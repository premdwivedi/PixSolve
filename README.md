# PixSolve
PixSolve is a cutting-edge framework designed for the recognition and solution of mathematical and scientific expressions in real-world scenarios. This powerful Streamlit application accurately recognizes and converts mathematical equations from images and documents into text format. Recognizing complex math expressions is highly challenging due to the variety of symbols, notations, and layout complexities. PixSolve not only extracts these expressions but also provides their solutions and detailed information by integrating with the GPT-3.5 Turbo API. PixSolve provides an intuitive user interface with two modes: Direct Recognition for uploading images containing equations, and Manual Selection for uploading PDFs or images to manually crop/select equation regions. Leveraging cutting-edge deep learning, PixSolve detects individual symbols, numbers, and notation elements, then reconstructs full equations by understanding their spatial relationships. A key innovation is PixSolve's Length Awareness capability, dynamically allocating computational power to efficiently process expressions of varying complexities, from simple to highly intricate. Extensive experiments demonstrate that PixSolve significantly outperforms state-of-the-art methods in recognizing both printed and handwritten expressions. By leveraging the GPT-3.5 Turbo API, PixSolve goes beyond recognition to provide solutions and comprehensive information about the extracted expressions, making it a powerful tool for scientific and educational applications. This state-of-the-art Streamlit application enables digitizing mathematical content from diverse sources, allowing equations to be searchable, editable, and accessible across devices and platforms for various applications such as academic research, education, and document processing. PixSolve is publicly available, promoting further research and development in the field of expression recognition and solution.

# To use this Project
step 1: Clone the repo
Srep 2: cd models

#Download the model and tokenizer individually or use git-lfs

git lfs install

git clone https://huggingface.co/wanderkid/unimernet

# Installation 
conda create -n pixsolve python=3.12

conda activate pixsolve

pip install --upgrade unimernet

## Get your Gpt API Key

# RUN
Code: streamlit run pixolve_aiml.py