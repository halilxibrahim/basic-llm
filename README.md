# Basic LLM - Aviation and Defense Industry Dictionary

This project creates a specialized GPT-like language model using PyTorch, focused on aviation and defense industry terminology. The model learns technical terms and definitions to generate dictionary-like content.

## 🚀 Features

- **Transformer Architecture**: Modern GPT-like transformer model
- **Aviation Terminology**: Specialized in aviation and defense industry terms
- **Dictionary Generation**: AI capable of generating technical terms and definitions
- **Interactive Querying**: Term research based on user inputs
- **Quality Assessment**: Special metrics to measure model performance
- **GPU Support**: CUDA-enabled GPU acceleration

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Other required packages (listed in requirements.txt file)

## 🛠️ Installation

1. **Clone the project:**
   ```bash
   git clone <repository-url>
   cd basic-llm
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment:**
   ```bash
   # For macOS/Linux:
   source .venv/bin/activate
   
   # For Windows:
   .venv\Scripts\activate
   ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 1. Model Training

To train the aviation dictionary model, run the `train.py` file:

```bash
python train.py
```

This command:
- Reads aviation terms from the `input.txt` file
- Trains the model for 3000 iterations (optimized for aviation data)
- Saves the trained model as `gpt_model.pth`
- Shows loss values during training

### 2. Dictionary Text Generation

To generate aviation dictionary content with the trained model:

```bash
python dictionary_generate.py
```

This command:
- Loads the trained model
- Automatically generates 5 aviation terms and definitions
- Starts interactive query mode
- Generates AI definitions for user-entered terms

### 3. Model Quality Evaluation

To analyze model performance:

```bash
python evaluate_dictionary.py
```

This command:
- Evaluates model quality with special metrics
- Calculates term coverage ratio
- Analyzes technical terminology density
- Provides overall performance score

### 4. Traditional Text Generation

Still available for simple text generation:

```bash
python generate.py
```

## 📁 File Structure

```
basic-llm/
├── model.py                # GPT model architecture (optimized for aviation)
├── train.py               # Model training script
├── generate.py            # Traditional text generation
├── dictionary_generate.py # Aviation dictionary generation and interactive query
├── evaluate_dictionary.py # Model quality evaluation
├── input.txt              # Aviation and defense industry terms
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## ⚙️ Model Parameters

The model is configured with hyperparameters optimized for aviation terminology:

- **Batch Size**: 32 (optimized for technical terms)
- **Block Size**: 512 (extended context for long definitions)
- **Embedding Size**: 512 (richer term representation)
- **Attention Heads**: 8 (more detailed relationship modeling)
- **Layers**: 8 (deeper terminology learning)
- **Learning Rate**: 1e-4 (more careful learning for technical content)
- **Dropout**: 0.1 (reduced for limited data)

You can modify these parameters in the `model.py` file.

## 🎨 Customization

### Training with Different Data

To use your own specialized dictionary data:
1. Replace the `input.txt` file with your own terminology dataset
2. Follow the same format: "TERM: Definition - Description..."
3. Retrain the model: `python train.py`

### Modifying Model Parameters

You can adjust the model size and performance by changing the hyperparameters in `model.py`:

```python
n_embd = 512    # Embedding size (current: optimized for aviation terms)
n_head = 8      # Number of attention heads
n_layer = 8     # Number of transformer layers
block_size = 512 # Context window size
```

### Adding New Term Categories

To expand beyond aviation terms:
1. Add new technical terminology to `input.txt`
2. Update the evaluation keywords in `evaluate_dictionary.py`
3. Modify the starter terms in `dictionary_generate.py`

## 📊 Performance

Model performance depends on:
- **Quality and quantity** of specialized terminology data
- **Model architecture** (embedding size, layers, attention heads)
- **Training duration** (iterations and convergence)
- **Domain specificity** (focused vs. general vocabulary)

## 🐛 Troubleshooting

### GPU Usage
The model automatically uses GPU if available. To check GPU usage:
```python
print(f"Using device: {device}")
```

### Memory Issues
For large models or limited GPU memory, reduce batch size:
```python
batch_size = 16  # Instead of 32
```

### Training Time
To adjust training duration:
```python
max_iters = 1500  # Reduce for faster training
max_iters = 5000  # Increase for better quality
```

### Dictionary Quality
If generated content quality is low:
- Increase training iterations
- Add more diverse terminology data
- Adjust learning rate (try 5e-5 for slower, more careful learning)

## 🚁 Example Output

After training, the model can generate content like:

```
STEALTH: Stealth Technology - Advanced design techniques that reduce aircraft detectability by radar, infrared, and other detection methods. Modern military aircraft incorporate stealth features for tactical advantage.

VTOL: Vertical Take-Off and Landing - Aircraft capability to take off and land vertically without requiring a runway. Helicopters and specialized aircraft like the F-35B utilize VTOL technology.
```

## 🎓 Educational Use Cases

- **Aviation Training**: Learn technical terminology
- **Defense Studies**: Understand military technology concepts  
- **Language Learning**: Technical English vocabulary building
- **Research**: Terminology standardization and definition generation

## 🤝 Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature/new-terminology`)
3. Add new terms or improve existing functionality
4. Commit your changes (`git commit -m 'Add aerospace terminology'`)
5. Push to the branch (`git push origin feature/new-terminology`)
6. Create a Pull Request

## 📋 Future Enhancements

- [ ] Multi-language support (Turkish-English bilingual)
- [ ] Larger aviation terminology database
- [ ] Fine-tuning with specialized aerospace literature
- [ ] Integration with aviation knowledge bases
- [ ] Advanced evaluation metrics for domain accuracy

---

**Note**: This model is for educational and research purposes. Generated definitions should be verified with authoritative aviation sources for critical applications. 