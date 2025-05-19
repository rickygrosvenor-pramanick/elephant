
# ELEPHANT

This repository contains the data and code associated with the ELEPHANT paper. We provide code to get model responses to our prompt datasets, run the ELEPHANT social sycophancy metrics on model outputs from an LLM (`elephant.py`) and then compare these outputs to human responses (`compare_to_human.ipynb`).

## Setup
Clone this repository.
You will need to set your `OPENAI_API_KEY` either as an environment variable or in a separate file called `key.txt`.

## 📂 Data

We provide the following datasets:

- **Open-Ended Questions (OEQ)**  
  Full dataset of personal advice-seeking queries, including human responses rated on each ELEPHANT metric:  
  `datasets/OEQ.csv`

- **Am I The Asshole (AITA)**  
  Full dataset of AITA-style queries, including human responses and ground truth labels:  
  `datasets/AITA.csv`

### 🔍 Sample Datasets for Testing

To quickly test the pipeline, you can use the sample files (each with 10 examples):
- `datasets/OEQ_sample.csv`  
- `datasets/AITA_sample.csv`
These are smaller versions of the full datasets and useful for debugging or exploration.


## Steps to use ELEPHANT for a given model

### Step 0. Get LLM Outputs: 
First, you need to run inference on the model to get responses to prompts in **OEQ** and **AITA**. We provide sample code to get responses from GPT-4o in `get_responses_gpt.py`.
*This step can be skipped if you already have LLM outputs, or if you are evaluating a preference datasets where the outputs already exist*

### Step 1. Run ELEPHANT metrics: 
The `elephant.py` script computes **social sycophancy metrics**: either on open-ended data (OEQ), with four metrics:
  - *emotional validation*
  - *indirect language*
  - *indirect action*
  - *accepting framing*
or on AITA dataset for *moral endorsement*. 

Instead of OEQ, feel free to use any set of open-ended prompts instead of the OEQ dataset we provide.

#### 🔧 Usage

```bash
python elephant.py \
  --input_file <path_to_csv> \
  --prompt_column <column_with_prompts> \
  --response_column <column_with_model_responses> \
  [--output_column_tag <tag>] \
  [--output_file <path_to_save_csv>] \
  (--AITA | --OEQ)
```

#### Required arguments:
- `--input_file`: Path to input CSV file.
- `--prompt_column`: Column to use as input prompt.
- `--response_column`: Column to use as model output.
- `--AITA` or `--OEQ`: Specify which dataset to run metrics for. **Exactly one is required.**

#### Optional arguments:
- `--output_column_tag`: A tag used to name the output metric columns (e.g., `gpt4` → `emotional_validation_gpt4`)
- `--output_file`: Where to save the annotated CSV. If omitted, the file will be `input_file_elephant_scored.csv`

#### 📝 Output
New columns will be added to the CSV for each evaluated metric:
- For OEQ: `emotional_validation_<tag>`, `indirect_action_<tag>`, `indirect_language_<tag>`, `accept_framing_<tag>`
- For AITA: `moral_endorsement_<tag>`, `accept_framing_<tag>`, `emotional_validation_<tag>`
  
### Step 2. Compare to human responses
Next, you can use the `compare_to_human.ipynb` notebook to compare it with human responses. In that notebook, you can generate plots and results for rates of social sycophancy in models compared to humans. It is currently saved as an example on the 10-sample dataset, which we walk through below. 

# Example pipeline on small (10-example) subset of the data
## Step 0. Get inputs
For example, to get outputs on OEQ, AITA on the binary setting, and AITA in open-ended setting:
  ```bash
  python get_responses_gpt.py \
  --input_file datasets/OEQ_sample.csv \
  --input_column prompt \
  --output_file outputs/OEQ_responses.csv \
  --output_column gpt_response

  python get_responses_gpt.py \
  --input_file datasets/AITA_sample.csv \
  --input_column prompt \
  --output_file  outputs/AITA_responses.csv \
  --output_column gpt_response_binary --AITA_binary

  python get_responses_gpt.py \
  --input_file datasets/AITA_sample.csv 
  --input_column prompt \
  --output_file outputs/AITA_responses.csv \
  --output_column gpt_response
   ```

## Step 1. Run ELEPHANT metrics
##### Run on OEQ:
```bash
python elephant.py \
  --input_file outputs/OEQ_responses.csv \
  --prompt_column prompt \
  --response_column gpt_response \
  --output_column_tag gpt4o \
  --OEQ
```

##### Run on AITA:
```bash
python elephant.py \
  --input_file outputs/AITA_responses.csv \
  --prompt_column prompt \
  --response_column gpt_response \
  --output_column_tag gpt4o \
  --AITA
```

## Step 2. Compare to humans
We provide the example results and plots and `compare_to_human.ipynb`


# Additional data and code
### Full datasets
We provide the full datasets of responses from 8 different models on OEQ and AITA, and Python notebooks to analyze the results, in the `full_datasets_from_paper` folder

#### Mitigations for AITA: `get_responses_with_mitigations.py`
In the same folder, run `python get_responses_with_mitigations.py` to run the full pipeline for outputs from GPT-4o: getting binary responses, getting open-ended responses, and then running the **moral endorsement** metric.
