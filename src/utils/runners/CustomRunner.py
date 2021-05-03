from catalyst import dl


class CustomRunner(dl.Runner):
    def __init__(self, custom_metrics):
        super().__init__()
        self.custom_metrics = custom_metrics

    def handle_batch(self, batch):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `train()`.

        y_pred = self.model(**batch)  # Forward pass

        target_tag = batch['target_tag']
        attention_mask = batch['attention_mask']
        # Compute the loss value
        loss = self.criterion(
            y_pred,
            target_tag,
            attention_mask=attention_mask,
            output_dim=self.model.output_dim,
        )

        metrics_result = {'loss': loss}
        for name, metric_fn in self.custom_metrics.items():
            metrics_result[name] = metric_fn(y_pred, target_tag, attention_mask)

        # Update metrics (includes the metric that tracks the loss)
        self.batch_metrics.update(metrics_result)

        if self.is_train_loader:
            # Compute gradients
            loss.backward()
            # Update weights
            # (the optimizer is stored in `self.state`)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
