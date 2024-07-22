# Train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
k_folds = 5
num_epochs = 50

# Our dataset has two classes only - background and crater
num_classes = 2
# Use our dataset and defined transformations
dataset = CraterDataset('/kaggle/input/martianlunar-crater-detection-dataset/craters/train', get_transform(train=True))
dataset_val = CraterDataset('/kaggle/input/martianlunar-crater-detection-dataset/craters/train', get_transform(train=False))

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start print
print('--------------------------------')
# K-fold Cross Validation model evaluation
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    dataset_subset = Subset(dataset, list(train_ids))
    dataset_val_subset = Subset(dataset_val, list(val_ids))

    # Define training and validation data loaders
    data_loader = DataLoader(
        dataset_subset, batch_size=8, shuffle=True, num_workers=2,
        collate_fn=collate_fn
    )

    data_loader_val = DataLoader(
        dataset_val_subset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn
    )

    # Get the model using our helper function
    model = get_model_bbox(num_classes)
    
    # Move model to the right device
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0)

    # And a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Let's train!
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 50 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        # Update the learning rate
        lr_scheduler.step()
    
    # Evaluate on the validation dataset
    evaluate(model, data_loader_val, device=device)
