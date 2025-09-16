# VANGUARD: Variational Anatomical Neural Generator for Unified and Adaptive Reconstruction in Diverse imaging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A novel Bayesian deep learning approach for cross-site MRI harmonization combining uncertainty quantification, anatomical priors, and adaptive tissue-specific regularization for T2-weighted fetal brain MRI.

**Author**: Daniel Barreras ([DanielBrMz](https://github.com/DanielBrMz))

## Key Innovations

- **Bayesian Neural Architecture**: Variational inference with principled uncertainty quantification
- **Anatomical Prior Integration**: dHCP template-guided harmonization with tissue segmentation awareness  
- **Adaptive Regularization**: Dynamic loss weighting based on tissue-specific harmonization requirements
- **Clinical Validation**: Designed for multi-center fetal neuroimaging studies with traveling subject validation

## Quick Start

```bash
# Clone repository
git clone https://github.com/DanielBrMz/vanguard-mri-harmonization
cd vanguard-mri-harmonization

# Setup environment
conda env create -f environment.yml
conda activate vanguard

# Install package in development mode
pip install -e .

# Verify installation
python -c "import vanguard; print('VANGUARD successfully installed')"
```

## Project Status

🚧 **Under Active Development** - Core mathematical foundations being implemented by Daniel Barreras

## About the Author

Daniel Barreras is developing VANGUARD as part of advanced research in Bayesian deep learning applications to medical image harmonization. This work focuses on principled uncertainty quantification and anatomical prior integration for fetal brain MRI analysis.

## Documentation

- [Mathematical Foundation](docs/source/mathematical_foundation.rst)
- [Architecture Overview](docs/source/architecture.rst)
- [API Reference](docs/source/api/)

## Citation

```bibtex
@software{barreras2025vanguard,
    title={VANGUARD: Variational Anatomical Neural Generator for Unified and Adaptive Reconstruction in Diverse imaging},
    author={Barreras, Daniel},
    year={2025},
    url={https://github.com/DanielBrMz/vanguard-mri-harmonization}
}
```

## Contact

- **GitHub**: [@DanielBrMz](https://github.com/DanielBrMz)
- **Email**: Available through GitHub profile

## License

MIT License - see [LICENSE](LICENSE) file for details.
