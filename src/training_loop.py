
import sys
import contextlib
import numpy as np
import time, datetime

import torch
from torch.utils.data import TensorDataset, IterableDataset, DataLoader

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def resolve_device( device_str ):
    """Resolve 'auto' to best available device: mps -> cuda -> cpu."""
    if device_str != 'auto':
        return device_str
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def estimate_dataset_rows( dataset ):
    """Best-effort row estimate for progress reporting on parquet-backed datasets."""
    cached_rows = getattr( dataset, '_cached_num_rows', None )
    if cached_rows is not None:
        return cached_rows

    parquet_files = getattr( dataset, 'parquet_files', None )
    if pq is None or parquet_files is None:
        return None

    try:
        total_rows = 0
        for filepath in parquet_files:
            total_rows += pq.ParquetFile( filepath ).metadata.num_rows
        dataset._cached_num_rows = int( total_rows )
        return dataset._cached_num_rows
    except Exception:
        return None


def train_model( model,
                 datasets,
                 initial_batch_size,
                 max_batch_size,
                 epochs_to_double_batch,
                 loss_fx,
                 optimizer,
                 num_epochs,
                 train_device,
                 other_device,
                 file_name,
                 progress_tick_rows=0,
                 num_workers=0,
                 epoch_callback=None,
                 batch_callback=None,
                 checkpoint_metric_callback=None,
                 report_epoch_loss=True,
                 checkpoint_phase='test',
                 skip_batch_phases=None,
                 patience=None,
                 start_epoch=1, ):

    s_time = time.time()

    train_device = resolve_device( train_device )
    other_device = resolve_device( other_device )
    print( 'Train device: ' + train_device + ', eval device: ' + other_device )

    phases = list( datasets )
    skip_batch_phases = set( skip_batch_phases or [] )
    if len( skip_batch_phases ) > 0:
        invalid_phases = [ p for p in skip_batch_phases if p not in phases ]
        if len( invalid_phases ) > 0:
            raise ValueError( 'Undefined skip_batch_phases: ' + ', '.join( invalid_phases ) )
        if checkpoint_metric_callback is None:
            raise ValueError( 'skip_batch_phases requires checkpoint_metric_callback' )
        if any( p != checkpoint_phase for p in skip_batch_phases ):
            raise ValueError( 'skip_batch_phases only supports checkpoint_phase=' + str(checkpoint_phase) )

    # Detect if datasets are IterableDataset (e.g. parquet streaming)
    is_iterable = isinstance( datasets[ phases[0] ], IterableDataset )

    batch_sizes = dict( [ ( p, 0 ) if p == 'train'
                          else ( p, max_batch_size ) for p in phases ] )
    devices = dict( [ ( p, train_device ) if p == 'train'
                      else ( p, other_device ) for p in phases ] )

    best_epoch = 0
    best_loss = 1e40
    tolerance = 1e-4
    epochs_wo_improv = 0

    for epoch in range( start_epoch, num_epochs+1 ):
        print( 'Epoch ' + str(epoch) + ' of ' + str(num_epochs) )
        print( '-' * 50 )

        # Call set_epoch for IterableDatasets (per-epoch shard shuffling)
        for p in phases:
            if hasattr( datasets[p], 'set_epoch' ):
                datasets[p].set_epoch( epoch )

        float_batch_scaler = initial_batch_size * np.exp( np.log(2) * (epoch-1) / epochs_to_double_batch  ) / 8
        train_batch_size = int( round( float_batch_scaler ) ) * 8
        eval_batch_size = min( max_batch_size, train_batch_size )
        for p in phases:
            if p != 'train':
                batch_sizes[p] = eval_batch_size
        if train_batch_size != batch_sizes['train'] or is_iterable:
            batch_sizes['train'] = train_batch_size
            if is_iterable:
                dataloaders = dict( [ ( p, None ) if p in skip_batch_phases
                                      else ( p, DataLoader( datasets[p], batch_sizes[p],
                                                            shuffle=False, num_workers=num_workers, ) )
                                    for p in phases ] )
            else:
                dataloaders = dict( [ ( p, None ) if p in skip_batch_phases
                                      else ( p, DataLoader( datasets[p], batch_sizes[p], shuffle=True, ) )
                                    for p in phases ] )

        for phase in phases:
            model.to( devices[ phase ] )
            loss_fx.to( devices[ phase ] )

            if phase == 'train':
                model.train()  # Set model to training mode
                print( 'Batch size = ' + str(batch_sizes['train']) )
            else:
                model.eval()   # Set model to evaluate mode
                estimated_rows = estimate_dataset_rows( datasets[ phase ] )
                progress_label = phase.capitalize() + ' progress'
                if estimated_rows is not None:
                    print( phase.capitalize() + ' batch size = ' + str(batch_sizes[ phase ]) +
                           ', estimated rows = ' + str(estimated_rows) )
                else:
                    print( phase.capitalize() + ' batch size = ' + str(batch_sizes[ phase ]) )

            running_loss = 0.0
            total_samples = 0
            tick_rows_accum = 0
            tick_count = 0
            next_phase_progress_rows = progress_tick_rows if progress_tick_rows > 0 else None
            skip_phase_batches = phase in skip_batch_phases

            data = dataloaders[phase]
            grad_context = contextlib.nullcontext() if phase == 'train' else torch.inference_mode()

            # Iterate over data.
            if skip_phase_batches:
                print( phase.capitalize() + ' batches delegated to checkpoint callback' )
            else:
                with grad_context:
                    for i, batch in enumerate( data ):
                        batch_size = batch[0].size(0)
                        batch = [ b.to( devices[phase] ) for b in batch ]
                        inputs = batch[:-2]
                        outputs = batch[-2:] # y and weight/source

                        pred = model( *inputs )
                        loss = loss_fx( pred, *outputs, )

                        if phase == 'train':
                            # zero the parameter gradients
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # Progress ticks
                            if progress_tick_rows > 0:
                                tick_rows_accum += batch_size
                                while tick_rows_accum >= progress_tick_rows:
                                    tick_rows_accum -= progress_tick_rows
                                    tick_count += 1
                                    sys.stdout.write( '.' )
                                    if tick_count % 10 == 0:
                                        sys.stdout.write( ' ' )
                                    sys.stdout.flush()

                            if batch_callback is not None:
                                batch_callback( model=model,
                                                epoch=epoch,
                                                phase=phase,
                                                batch_index=i + 1,
                                                batch_size=batch_size,
                                                batch_loss=float( loss.item() ),
                                                device=devices[ phase ] )

                        # statistics
                        if report_epoch_loss:
                            running_loss += loss.item() * batch_size
                        total_samples += batch_size

                        if phase != 'train' and next_phase_progress_rows is not None:
                            while total_samples >= next_phase_progress_rows:
                                if estimated_rows is not None and estimated_rows > 0:
                                    percent = min( 100.0, 100.0 * total_samples / estimated_rows )
                                    print( progress_label + ': ' +
                                           format( percent, '6.2f' ) + '% (' +
                                           str(total_samples) + '/' + str(estimated_rows) + ' rows)' )
                                else:
                                    print( progress_label + ': ' + str(total_samples) + ' rows processed' )
                                next_phase_progress_rows += progress_tick_rows

            # End of phase newline after ticks
            if phase == 'train' and progress_tick_rows > 0 and tick_count > 0:
                print()
            elif phase != 'train' and progress_tick_rows > 0 and total_samples > 0:
                if estimated_rows is not None:
                    percent = min( 100.0, 100.0 * total_samples / estimated_rows ) if estimated_rows > 0 else 100.0
                    print( progress_label + ': ' + format( percent, '6.2f' ) + '% (' + str(total_samples) +
                           '/' + str(estimated_rows) + ' rows)' )
                else:
                    print( progress_label + ': ' + str(total_samples) + ' rows processed' )

            epoch_loss = None
            if report_epoch_loss and total_samples > 0:
                epoch_loss = running_loss / total_samples
            runtime = time.time() - s_time
            if report_epoch_loss and epoch_loss is not None:
                print( phase.capitalize() + format( epoch_loss, '.4f' ).rjust(8) )

            checkpoint_loss = epoch_loss
            checkpoint_label = 'loss'
            if phase == checkpoint_phase and checkpoint_metric_callback is not None:
                checkpoint_loss, checkpoint_label = checkpoint_metric_callback( model=model,
                                                                               dataset=datasets[ phase ],
                                                                               phase=phase,
                                                                               device=devices[ phase ],
                                                                               epoch=epoch,
                                                                               epoch_loss=epoch_loss )
                print( 'Checkpoint ' + checkpoint_label + ': ' + format( float(checkpoint_loss), '.6f' ) )

            if phase == checkpoint_phase:
                #MAEs = loss_fx.source_b.weight.cpu().detach().numpy().tolist()[0]
                #for t, learned_mae in enumerate( MAEs ):
                #    print( '\t' + unique_sources[t].ljust(25) + format(learned_mae,'.3f') )
                if checkpoint_loss < best_loss-tolerance:
                    print("New best weights! Copying and saving model")
                    best_epoch = epoch
                    best_loss = checkpoint_loss
                    torch.save( model.state_dict(), file_name )
                    epochs_wo_improv = 0
                else:
                    epochs_wo_improv += 1
                    print( 'Did not improve, best performance was epoch ' +
                           str(best_epoch) + ' (' + format(best_loss,'.4f') + ')' )
        if epoch_callback is not None:
            epoch_callback( model, epoch )

        runtime = time.time() - s_time
        print( 'Runtime: ' + str(datetime.timedelta(seconds=runtime)).split('.')[0] + '\n' )

        if patience is not None and epochs_wo_improv > patience:
            print( 'Early stopping: no improvement for ' + str(epochs_wo_improv) +
                   ' epochs (patience=' + str(patience) + ')' )
            break

    return best_loss


