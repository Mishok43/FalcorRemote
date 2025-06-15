from base import PSSampler, RandomSample, DummySampler
from typing import List, Dict, Any, Optional
import numpy as np
from typing_extensions import override
import torch
import time

# Dummy implementations for logging (removed dependencies)
class LogCategory:
    EVALUATION = "evaluation"

def get_logger():
    class DummyLogger:
        def info(self, msg, category=None):
            pass  # Silent logging for efficiency
    return DummyLogger()

try:
    import normflows as nf
    from normflows.distributions.base import BaseDistribution
    NORMFLOWS_AVAILABLE = True
except ImportError:
    NORMFLOWS_AVAILABLE = False
    # Create dummy base class for when normflows is not available
    class BaseDistribution:
        def __init__(self):
            pass
        def register_buffer(self, name, tensor):
            setattr(self, name, tensor)


class NFUniform(BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, shape, low=0.0, high=1.0):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          low: Lower bound of uniform distribution
          high: Upper bound of uniform distribution
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.d = np.prod(shape)
        self.register_buffer("low", torch.tensor(low).expand(self.shape))
        self.register_buffer("high", torch.tensor(high).expand(self.shape))

    def forward(self, num_samples=1, context=None):
        # Create the shape tuple for torch.rand
        sample_shape = (num_samples,) + self.shape
        eps = torch.rand(sample_shape, dtype=self.low.dtype, device=self.low.device)
        z = self.low + (self.high - self.low) * eps

        log_p = -self.d * torch.log(self.high[0] - self.low[0] ) * torch.ones(num_samples, device=self.low.device)

        return z, log_p

    def log_prob(self, z, context=None):
        log_p = -self.d * torch.log(self.high[0] - self.low[0]) * torch.ones(z.shape[0], device=z.device)
        out_range = torch.logical_or(z < self.low, z > self.high)
        ind_inf = torch.any(torch.reshape(out_range, (z.shape[0], -1)), dim=-1)
        log_p[ind_inf] = -torch.inf
        return log_p


class NFSampler(DummySampler):
    """
    Normalizing Flow-based sampler for importance sampling.

    This sampler learns from observations (x, y) where x ∈ [0,1]^d and y is a performance metric.
    It uses normalizing flows to learn a complex distribution that focuses sampling on
    regions with better performance (lower y values).

    Usage:
        sampler = NFSampler(name="nf_sampler", rng=None, num_flows=4, hidden_units=128)
        # Register dimensions first
        for dist in distributions:
            dist.register_with_sampler(sampler)
        sampler.reinitialize()  # Initialize the neural network

        # Add training data
        sampler.add_data(x_data, x_pdf_data, y_data)
        sampler.fit()  # Train the model

        # Sample
        sample = sampler.sample_primary()  # RandomSample with learned PDF
    """

    def __init__(self, name=None, rng=None, num_flows=4, hidden_units=128, hidden_layers=3,
                 learning_rate=1e-3, epochs_per_fit=25, batch_size=100, history_size=300, latent_size=5, device=None):
        super().__init__(name, rng)

        if not NORMFLOWS_AVAILABLE:
            raise ImportError("normflows package is required for NFSampler. Install with: pip install normflows")

        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.num_flows = num_flows
        self.is_initialized = False
        self.epochs_per_fit = epochs_per_fit
        self.has_dummy_dimension = False
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.history_size = history_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger()
        # Initialize concatenated data storage - keep unnormalized y values
        self.all_x = None
        self.all_x_pdf = None
        self.all_y_unnormalized = None  # Keep original unnormalized values

    @override
    def reinitialize(self):
        """Initialize the normalizing flow model after dimensions are registered"""
        if self.is_initialized:
            raise Exception("NFSampler is already initialized")

        if self.total_dimensions == 0:
            raise Exception("No dimensions registered. Call register_dimension() first.")

        torch.manual_seed(0)


        flows = []
        # Use coupling layers for all cases since we ensure at least 2D
        for i in range(self.num_flows):
            flows += [nf.flows.CoupledRationalQuadraticSpline(self.latent_size, self.hidden_layers,
                                                            self.hidden_units, tails=None, tail_bound=1.0)]
            flows += [nf.flows.Permute(self.latent_size, mode='swap')]

        self.logger.info(f"NFSampler initialized with {self.latent_size} dimensions", category=LogCategory.EVALUATION)

        q0 = NFUniform(self.latent_size, 0.0, 1.0)
        self.model = nf.NormalizingFlow(q0=q0, flows=flows, p=None)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.is_initialized = True

    def visualize(self):


        ax1.pcolormesh(xx, yy, prob_target.data.numpy())
        ax1.set_aspect('equal', 'box')
        ax1.set_title('Target Distribution')

        ax2.pcolormesh(xx, yy, prob.data.numpy())
        ax2.set_aspect('equal', 'box')
        ax2.set_title('Current Distribution')

        plt.show()

    def add_data(self, x: List[List[float]], x_pdf: List[float], y: List[float], alpha: float = 0.5):
        if True:
            """
            Add data to history as pytorch tensors and concatenate immediately

            Args:
                x: List of multi-dimensional input vectors in [0,1]^d
                x_pdf: List of PDF values for each x sample
                y: List of performance values (lower is better)
                alpha: Learning rate (not used in this implementation, kept for compatibility)
            """
            if not self.is_initialized:
                raise Exception("NFSampler must be initialized before adding data. Call reinitialize() first.")

            # Convert to tensors
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)

            # Add dummy dimension if needed for 1D case
            if self.has_dummy_dimension:
                dummy_dim = torch.ones(x_tensor.shape[0], 1, device=self.device)
                x_tensor = torch.cat([x_tensor, dummy_dim], dim=1)

            x_pdf_tensor = torch.tensor(x_pdf, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

            # Concatenate with existing data or initialize
            if self.all_x is None:
                self.all_x = x_tensor
                self.all_x_pdf = x_pdf_tensor
                self.all_y_unnormalized = y_tensor
            else:
                self.all_x = torch.cat([self.all_x, x_tensor], dim=0)
                self.all_x_pdf = torch.cat([self.all_x_pdf, x_pdf_tensor], dim=0)
                self.all_y_unnormalized = torch.cat([self.all_y_unnormalized, y_tensor], dim=0)

            # Keep only the most recent data points
            if self.all_x.shape[0] > self.history_size:
                self.all_x = self.all_x[-self.history_size:]
                self.all_x_pdf = self.all_x_pdf[-self.history_size:]
                self.all_y_unnormalized = self.all_y_unnormalized[-self.history_size:]

    def get_normalized_y(self):
        """Get normalized y values for training"""
        if self.all_y_unnormalized is None:
            return None

        # Calculate normalization constant from all unnormalized data
        y_norm_const = torch.mean(self.all_y_unnormalized / self.all_x_pdf)

        # Return normalized y values without modifying the original data
        return self.all_y_unnormalized / y_norm_const

    def fit(self):
        if True:
            """
            Fit the distribution to observed data points using gradient descent.

            Uses the loss function from real_nvp.ipynb:
            weight = -exp(log_p_pdf) / exp(log_pdf_q_sample)
            loss = mean(weight * log_pdf_q)
            """
            if not self.is_initialized:
                raise Exception("NFSampler must be initialized before fitting. Call reinitialize() first.")

            if self.all_x is None or self.all_x.shape[0] == 0:
                raise Exception("No data available for fitting. Call add_data() first.")

            # Get normalized y values for training
            normalized_y = self.get_normalized_y()
            if normalized_y is None:
                raise Exception("No normalized y data available for fitting.")

            n_samples = self.all_x.shape[0]
            if n_samples < self.batch_size:
                print(f"Warning: Only {n_samples} samples available, using batch size of {n_samples}")
                batch_size = n_samples
            else:
                batch_size = self.batch_size

            for epoch in range(self.epochs_per_fit):
                self.optimizer.zero_grad()

                # Randomly sample batch indices
                indices = torch.randint(0, n_samples, (batch_size,), device=self.device)
                x_batch = self.all_x[indices]
                x_pdf_batch = self.all_x_pdf[indices]
                y_batch = normalized_y[indices]

                # Calculate weight as in real_nvp.ipynb
                # weight = -exp(log_p_pdf) / exp(log_pdf_q_sample)
                weight = -y_batch / (x_pdf_batch + 1e-8)

                # Get log probability from our model
                log_pdf_q = self.model.log_prob(x_batch)

                # Calculate loss
                loss = torch.mean(weight * log_pdf_q)


                print(f"Epoch {epoch}/{self.epochs_per_fit}, Loss: {loss.item():.6f}")

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
                    self.optimizer.step()
                else:
                    print(f"NaN or Inf loss at epoch {epoch}, skipping optimization step")

    @override
    def sample(self, num_samples: int = 1) -> float:
        """Sample all registered dimensions and return total PDF for single sample"""
        if not self.is_initialized:
            raise Exception("NFSampler must be initialized before sampling. Call reinitialize() first.")

        if self.total_dimensions == 0:
            raise Exception("No dimensions registered with NFSampler")

        with torch.no_grad():
            x, log_pdf = self.model.sample(num_samples=1)  # Always sample 1 for this method
            x = x[0].cpu().numpy()  # Remove batch dimension and convert to numpy
            pdf = torch.exp(log_pdf[0]).cpu().numpy()  # Convert log_pdf to pdf

        # Remove dummy dimension if it was added
        if self.has_dummy_dimension:
            x = x[:-1]  # Remove the last dimension (dummy)

        # Store values for get() calls
        self.sampled_values = x.tolist()
        self.reset_counter()

        return pdf

    @override
    def sample_primary(self, num_samples: int = 1) -> List[RandomSample]:
        if True:
            """Sample using the learned normalizing flow model"""
            if not self.is_initialized:
                raise Exception("NFSampler must be initialized before sampling. Call reinitialize() first.")

            with torch.no_grad():
                x, log_pdf = self.model.sample(num_samples=num_samples)

            # Convert to numpy arrays
            x = x.cpu().numpy()
            pdfs = torch.exp(log_pdf).cpu().numpy()

            # Create list of RandomSample objects for all samples
            samples = []
            for i in range(num_samples):
                sample_x = x[i]

                # Remove dummy dimension if it was added
                if self.has_dummy_dimension:
                    sample_x = sample_x[:-1]  # Remove the last dimension (dummy)

                # Return full multidimensional value as list, or scalar for 1D
                value = sample_x.tolist()
                samples.append(RandomSample(value, pdfs[i]))

            # Always return list
            return samples

    @override
    def to_dict(self) -> Dict[str, Any]:
        """Serialize sampler state for distributed computing"""
        state_dict = {
            "type": "NFSampler",
            "total_dimensions": self.total_dimensions,
            "hidden_units": self.hidden_units,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "num_flows": self.num_flows,
            "epochs_per_fit": self.epochs_per_fit,
            "batch_size": self.batch_size,
            "history_size": self.history_size,
            "device": str(self.device),
            "is_initialized": self.is_initialized
        }

        if self.is_initialized:
            state_dict["model_state_dict"] = self.model.state_dict()
            state_dict["optimizer_state_dict"] = self.optimizer.state_dict()

        return state_dict

    @classmethod
    @override
    def from_dict(cls, data: Dict[str, Any]) -> "NFSampler":
        """Reconstruct sampler from serialized state"""
        sampler = cls(
            name=None,
            rng=None,
            num_flows=data["num_flows"],
            hidden_units=data["hidden_units"],
            hidden_layers=data["hidden_layers"],
            learning_rate=data["learning_rate"],
            epochs_per_fit=data["epochs_per_fit"],
            batch_size=data["batch_size"],
            history_size=data["history_size"],
            device=data["device"]
        )

        sampler.total_dimensions = data["total_dimensions"]

        if data["is_initialized"]:
            sampler.reinitialize()
            sampler.model.load_state_dict(data["model_state_dict"])
            sampler.optimizer.load_state_dict(data["optimizer_state_dict"])

        return sampler

    def visualize(self, num_viz_samples: int = 2000, figsize: tuple = (15, 5)):
        """
        Visualize the learned distribution compared to historical data.

        Args:
            num_viz_samples: Number of samples to generate for visualization
            figsize: Figure size for the plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn are required for visualization")
            return

        if not self.is_initialized:
            print("NFSampler must be initialized before visualization")
            return

        if self.all_x is None or self.all_x.shape[0] == 0:
            print("No historical data available for visualization")
            return

        # Get historical data (remove dummy dimension if present)
        hist_x = self.all_x.cpu().numpy()
        hist_y = self.get_normalized_y().cpu().numpy()

        if self.has_dummy_dimension:
            hist_x = hist_x[:, :-1]  # Remove dummy dimension

        # Generate samples from learned distribution
        with torch.no_grad():
            learned_samples, learned_log_pdf = self.model.sample(num_samples=num_viz_samples)
            learned_samples = learned_samples.cpu().numpy()
            learned_pdf = torch.exp(learned_log_pdf).cpu().numpy()

        if self.has_dummy_dimension:
            learned_samples = learned_samples[:, :-1]  # Remove dummy dimension

        actual_dims = self.total_dimensions

        if actual_dims == 1:
            # 1D visualization
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            # Plot 1: Historical data scatter (x vs performance y)
            scatter = axes[0].scatter(hist_x.flatten(), hist_y, c=hist_y, cmap='viridis_r', alpha=0.6, s=20)
            axes[0].set_xlabel('Parameter Value')
            axes[0].set_ylabel('Performance (lower is better)')
            axes[0].set_title('Historical Data\n(Color = Performance)')
            plt.colorbar(scatter, ax=axes[0])

            # Plot 2: Distribution comparison
            axes[1].hist(hist_x.flatten(), bins=50, alpha=0.6, density=True, label='Historical Data', color='blue')
            axes[1].hist(learned_samples.flatten(), bins=50, alpha=0.6, density=True, label='Learned Distribution', color='red')
            axes[1].set_xlabel('Parameter Value')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Distribution Comparison')
            axes[1].legend()

            # Plot 3: Learned distribution weighted by PDF
            # Create bins and compute average performance per bin
            bins = np.linspace(0, 1, 30)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Bin the learned samples and compute average PDF per bin
            bin_indices = np.digitize(learned_samples.flatten(), bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)

            bin_pdfs = np.zeros(len(bin_centers))
            bin_counts = np.zeros(len(bin_centers))

            for i, pdf_val in zip(bin_indices, learned_pdf):
                bin_pdfs[i] += pdf_val
                bin_counts[i] += 1

            # Average PDF per bin (avoid division by zero)
            avg_bin_pdfs = np.where(bin_counts > 0, bin_pdfs / bin_counts, 0)

            axes[2].bar(bin_centers, avg_bin_pdfs, width=bins[1]-bins[0], alpha=0.7, color='red')
            axes[2].set_xlabel('Parameter Value')
            axes[2].set_ylabel('Average Learned PDF')
            axes[2].set_title('Learned Distribution Focus\n(Higher = More Likely to Sample)')

        elif actual_dims == 2:
            # 2D visualization
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            # Plot 1: Historical data scatter (colored by performance)
            scatter = axes[0].scatter(hist_x[:, 0], hist_x[:, 1], c=hist_y, cmap='viridis_r', alpha=0.6, s=20)
            axes[0].set_xlabel('Parameter 1')
            axes[0].set_ylabel('Parameter 2')
            axes[0].set_title('Historical Data\n(Color = Performance)')
            plt.colorbar(scatter, ax=axes[0])

            # Plot 2: Learned distribution samples
            axes[1].scatter(learned_samples[:, 0], learned_samples[:, 1], alpha=0.3, s=10, color='red')
            axes[1].set_xlabel('Parameter 1')
            axes[1].set_ylabel('Parameter 2')
            axes[1].set_title('Learned Distribution Samples')
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)

            # Plot 3: PDF heatmap
            from scipy.stats import gaussian_kde

            # Create a grid for the heatmap
            x_grid = np.linspace(0, 1, 50)
            y_grid = np.linspace(0, 1, 50)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

            # Evaluate learned distribution on the grid
            with torch.no_grad():
                grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=self.device)
                if self.has_dummy_dimension:
                    dummy_dim = torch.ones(grid_tensor.shape[0], 1, device=self.device)
                    grid_tensor = torch.cat([grid_tensor, dummy_dim], dim=1)

                grid_log_pdf = self.model.log_prob(grid_tensor).cpu().numpy()
                grid_pdf = np.exp(grid_log_pdf).reshape(X_grid.shape)

            contour = axes[2].contourf(X_grid, Y_grid, grid_pdf, levels=20, cmap='Reds')
            axes[2].set_xlabel('Parameter 1')
            axes[2].set_ylabel('Parameter 2')
            axes[2].set_title('Learned PDF Heatmap')
            plt.colorbar(contour, ax=axes[2])

        else:
            # Higher dimensional - just show some projections
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot projections onto first two dimensions
            axes[0].scatter(hist_x[:, 0], hist_x[:, 1], c=hist_y, cmap='viridis_r', alpha=0.6, s=20)
            axes[0].set_xlabel('Parameter 1')
            axes[0].set_ylabel('Parameter 2')
            axes[0].set_title(f'Historical Data (Dims 1-2)\n{actual_dims}D Problem')

            axes[1].scatter(learned_samples[:, 0], learned_samples[:, 1], alpha=0.3, s=10, color='red')
            axes[1].set_xlabel('Parameter 1')
            axes[1].set_ylabel('Parameter 2')
            axes[1].set_title('Learned Distribution (Dims 1-2)')
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

        # Print some statistics
        print(f"\n=== NFSampler Visualization Statistics ===")
        print(f"Historical data points: {len(hist_x)}")
        print(f"Actual dimensions: {actual_dims}")
        print(f"Performance range: [{hist_y.min():.4f}, {hist_y.max():.4f}]")
        print(f"Historical data range: [{hist_x.min():.4f}, {hist_x.max():.4f}]")
        print(f"Learned samples range: [{learned_samples.min():.4f}, {learned_samples.max():.4f}]")
        print(f"Learned PDF range: [{learned_pdf.min():.6f}, {learned_pdf.max():.6f}]")

    def get_likelihood(self, x: List[float]) -> float:
        """
        Get the likelihood of a given input under the learned distribution

        Args:
            x: Input vector in [0,1]^d

        Returns:
            float: Likelihood value (higher = more likely)
        """
        if not self.is_initialized:
            return 0.0

        try:
            # Convert to tensor
            x_tensor = torch.tensor([x], dtype=torch.float32, device=self.device)

            # Add dummy dimension if needed
            if self.has_dummy_dimension:
                dummy_dim = torch.ones(x_tensor.shape[0], 1, device=self.device)
                x_tensor = torch.cat([x_tensor, dummy_dim], dim=1)

            # Get log probability from the model
            with torch.no_grad():
                log_prob = self.model.log_prob(x_tensor)
                likelihood = torch.exp(log_prob).item()

            return likelihood

        except Exception as e:
            # Return 0 if evaluation fails
            return 0.0

    def save_weights(self, filepath: str):
        """
        Save the NFSampler model weights and training data to a file

        Args:
            filepath: Path to save the weights (should end with .pth or .pt)
        """
        if not self.is_initialized:
            raise Exception("NFSampler must be initialized before saving weights")

        # Prepare data to save
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'total_dimensions': self.total_dimensions,
                'latent_size': self.latent_size,
                'hidden_units': self.hidden_units,
                'hidden_layers': self.hidden_layers,
                'learning_rate': self.learning_rate,
                'num_flows': self.num_flows,
                'epochs_per_fit': self.epochs_per_fit,
                'batch_size': self.batch_size,
                'history_size': self.history_size,
                'has_dummy_dimension': self.has_dummy_dimension,
                'device': str(self.device)
            },
            'training_data': {
                'all_x': self.all_x.cpu() if self.all_x is not None else None,
                'all_x_pdf': self.all_x_pdf.cpu() if self.all_x_pdf is not None else None,
                'all_y_unnormalized': self.all_y_unnormalized.cpu() if self.all_y_unnormalized is not None else None
            },
            'metadata': {
                'save_timestamp': torch.tensor(time.time()),
                'normflows_version': getattr(nf, '__version__', 'unknown'),
                'torch_version': torch.__version__
            }
        }

        # Save to file
        torch.save(save_data, filepath)
        print(f"NFSampler weights and data saved to: {filepath}")

    def load_weights(self, filepath: str, load_training_data: bool = True):
        """
        Load NFSampler model weights and optionally training data from a file

        Args:
            filepath: Path to the saved weights file
            load_training_data: Whether to load the training data as well
        """
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")

        # Load data from file
        save_data = torch.load(filepath, map_location=self.device, weights_only=False)

        # Verify compatibility
        config = save_data['config']
        if config['total_dimensions'] != self.total_dimensions:
            raise ValueError(f"Dimension mismatch: saved model has {config['total_dimensions']} dimensions, "
                           f"current sampler has {self.total_dimensions} dimensions")

        if config['latent_size'] != self.latent_size:
            raise ValueError(f"Latent size mismatch: saved model has {config['latent_size']}, "
                           f"current sampler has {self.latent_size}")

        # Update configuration to match saved model
        self.hidden_units = config['hidden_units']
        self.hidden_layers = config['hidden_layers']
        self.learning_rate = config['learning_rate']
        self.num_flows = config['num_flows']
        self.epochs_per_fit = config['epochs_per_fit']
        self.batch_size = config['batch_size']
        self.history_size = config['history_size']
        self.has_dummy_dimension = config['has_dummy_dimension']

        # Reinitialize if not already done
        if not self.is_initialized:
            self.reinitialize()

        # Load model and optimizer states
        self.model.load_state_dict(save_data['model_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])

        # Load training data if requested
        if load_training_data and save_data['training_data']['all_x'] is not None:
            self.all_x = save_data['training_data']['all_x'].to(self.device)
            self.all_x_pdf = save_data['training_data']['all_x_pdf'].to(self.device)
            self.all_y_unnormalized = save_data['training_data']['all_y_unnormalized'].to(self.device)
            print(f"Loaded {self.all_x.shape[0]} training samples")
        else:
            print("Training data not loaded (either not requested or not available)")

        # Print metadata
        metadata = save_data.get('metadata', {})
        if 'save_timestamp' in metadata:
            import datetime
            save_time = datetime.datetime.fromtimestamp(metadata['save_timestamp'].item())
            print(f"Model saved on: {save_time}")

        print(f"NFSampler weights loaded from: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str, device: str = None) -> "NFSampler":
        """
        Create a new NFSampler instance and load weights from file

        Args:
            filepath: Path to the saved weights file
            device: Device to load the model on (if None, uses saved device)

        Returns:
            NFSampler: New instance with loaded weights
        """
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")

        # Load configuration from file
        save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        config = save_data['config']

        # Use saved device if not specified
        if device is None:
            device = config['device']

        # Create new sampler with saved configuration
        sampler = cls(
            name=None,
            rng=None,
            num_flows=config['num_flows'],
            hidden_units=config['hidden_units'],
            hidden_layers=config['hidden_layers'],
            learning_rate=config['learning_rate'],
            epochs_per_fit=config['epochs_per_fit'],
            batch_size=config['batch_size'],
            history_size=config['history_size'],
            latent_size=config['latent_size'],
            device=device
        )

        # Set dimensions and initialize
        sampler.total_dimensions = config['total_dimensions']
        sampler.has_dummy_dimension = config['has_dummy_dimension']

        # No need to register dimensions since we're setting total_dimensions directly
        # The dimensions are already accounted for in the saved config

        # Load weights
        sampler.load_weights(filepath, load_training_data=True)

        return sampler
