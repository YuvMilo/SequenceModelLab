from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, dl, criterion, optimizer, num_epochs, logger, plot=True):

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        # Loop over the data in the dataloader
        for input_data, labels in dl:
            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_data)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
        logger.log(epoch_loss / len(dl), None, epoch)

        if plot and (epoch % 10 == 0):
            plt.close("all")
            plt.plot(output[0, :300].detach().numpy(), color="orange")
            plt.plot(labels[0, :300].detach().numpy(), color="red")
            plt.plot(input_data[0, :300].detach().numpy(), color="blue")
            plt.show()
            pass
