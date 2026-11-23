from pathlib import Path
import pandas as pd
import torch
from torch import nn
from .utility import save_json


def train_model(model, loaders, device, args, output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion=nn.SmoothL1Loss(beta=1.0) if args.loss=='huber' else nn.MSELoss()
    history=[]; best=float('inf'); stale=0; best_epoch=-1
    model=model.to(device)
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch_idx,(x,tf,y,_) in enumerate(loaders['train']):
            if args.max_train_batches and batch_idx >= args.max_train_batches: break
            x=x.to(device);tf=tf.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output=model(x,tf)
            loss=criterion(output,y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss: check learning rate, data, normalization.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            optimizer.step();losses.append(loss.item())
        model.eval(); val_mse=0.;n=0
        with torch.no_grad():
            for batch_idx,(x,tf,y,_) in enumerate(loaders['val']):
                if args.max_val_batches and batch_idx >= args.max_val_batches: break
                p=model(x.to(device),tf.to(device))
                val_mse+=nn.functional.mse_loss(p,y.to(device),reduction='sum').item()
                n+=y.numel()
        score=val_mse/max(n,1)
        train_loss=sum(losses)/len(losses)
        history.append({'epoch':epoch,'train_loss':train_loss,'val_mse_scaled':score})
        print(f'[{args.model}] epoch {epoch:03d}: train_loss={train_loss:.5f} val_mse={score:.5f}',flush=True)
        if score<best-1e-7:
            best=score;stale=0;best_epoch=epoch
            state={'state_dict':model.state_dict(),'config':vars(args),
                   'best_val_mse_scaled':float(best),'best_epoch':epoch}
            torch.save(state,out/'best.pt')
        else:
            stale+=1
        if stale>=args.patience:
            print(f'Early stopping after {epoch} epochs')
            break
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot([r['epoch'] for r in history], [r['val_mse_scaled'] for r in history], label='Validation MSE')
        ax.set(xlabel='Epoch',ylabel='Normalized MSE',title='Validation learning curve')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(out/'learning_curve.png',dpi=160);plt.close(fig)
    except ImportError:
        pass
    save_json({'best_epoch':best_epoch,'best_val_mse_scaled':float(best)},out/'training_summary.json')
