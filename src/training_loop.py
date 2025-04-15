
import numpy as np
import time, datetime

import torch
from torch.utils.data import TensorDataset, DataLoader


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
                 file_name, ):
    
    s_time = time.time()
    
    phases = list( datasets )
    
    batch_sizes = dict( [ ( p, 0 ) if p == 'train' 
                          else ( p, max_batch_size ) for p in phases ] )
    devices = dict( [ ( p, train_device ) if p == 'train' 
                      else ( p, other_device ) for p in phases ] )
    
    best_epoch = 0
    best_loss = 1e40
    tolerance = 1e-4

    for epoch in range( 1, num_epochs+1 ):
        print( 'Epoch ' + str(epoch) + ' of ' + str(num_epochs) )
        print( '-' * 50 )
        
        float_batch_scaler = initial_batch_size * np.exp( np.log(2) * (epoch-1) / epochs_to_double_batch  ) / 8
        train_batch_size = int( round( float_batch_scaler ) ) * 8
        if train_batch_size != batch_sizes['train']:
            batch_sizes['train'] = train_batch_size
            dataloaders = dict( [ ( p, DataLoader( datasets[p], batch_sizes[p], shuffle=True, ) ) 
                                for p in phases ] )
            
        for phase in phases:
            model.to( devices[ phase ] )
            loss_fx.to( devices[ phase ] )
            
            if phase == 'train':
                model.train()  # Set model to training mode
                print( 'Batch size = ' + str(batch_sizes['train']) )
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            total_samples = 0

            data = dataloaders[phase]

            # Iterate over data.
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
                    
                # statistics
                running_loss += loss.item() * batch_size
                total_samples += batch_size

            epoch_loss = running_loss / total_samples
            runtime = time.time() - s_time
            print( phase.capitalize() + format( epoch_loss, '.4f' ).rjust(8) )
            

            

            if phase == 'test':
                #MAEs = loss_fx.source_b.weight.cpu().detach().numpy().tolist()[0]
                #for t, learned_mae in enumerate( MAEs ):
                #    print( '\t' + unique_sources[t].ljust(25) + format(learned_mae,'.3f') )
                if epoch_loss < best_loss-tolerance:
                    print("New best weights! Copying and saving model")
                    best_epoch = epoch
                    best_loss = epoch_loss
                    torch.save( model.state_dict(), file_name )
                    epochs_wo_improv = 0
                else:
                    print( 'Did not improve, best performance was epoch ' + 
                           str(best_epoch) + ' (' + format(best_loss,'.4f') + ')' )
        runtime = time.time() - s_time
        print( 'Runtime: ' + str(datetime.timedelta(seconds=runtime)).split('.')[0] + '\n' )
    
    return best_loss



