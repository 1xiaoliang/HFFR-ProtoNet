# HFFR-ProtoNet
This is the official implementation of Hierarchical Feature Fusion Refinement Prototype Network (HFFR-ProtoNet).
## Installation

### Prerequisites

List the dependencies or environment required to run the project, for example:

- Python 3.9+
- ubuntu 22.04+

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/1xiaoliang/HFFR-ProtoNet.git
   ```
2. Installation
    ```bash
    cd HFFR-ProtoNet
    pip install -r requirements.txt
    ```
## Usage

If you need to perform pre-training, run the following command:
   ```bash
    python pretrain.py
   ```
If you want to perform meta-learning, run the following command:
   ```bash
    python meta_train.py
   ```
For model evaluation, use:
   ```bash
    python eval.py
   ```
The data_process.ipynb file contains some data processing methods for reference.
## License
This project is licensed under the MIT License.

