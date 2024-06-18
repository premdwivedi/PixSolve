import io
import os
import sys
import argparse
import numpy as np
import torch
import hashlib
import pypdfium2
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import re
from pylatexenc.latex2text import LatexNodes2Text
import xml.etree.ElementTree as ET

import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pixsolve.tasks as tasks
from pixsolve.common.config import Config
from pixsolve.processors import load_processor



MAX_WIDTH = 872
MAX_HEIGHT = 1024


class ImageProcessor:
    """ImageProcessor class handles the loading of the model and processing of images."""
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        # Load the model and visual processor from the configuration
        args = argparse.Namespace(cfg_path=self.cfg_path, options=None)
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg).to(self.device)
        vis_processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )
        return model, vis_processor

    def process_single_image(self, pil_image):
        # Process an image and return the LaTeX string
        image = self.vis_processor(pil_image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image})
        pred = output["pred_str"][0]
        return pred


@st.cache_data(show_spinner=False)
def read_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=300):
    # Extract an image from a PDF page
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image


@st.cache_data()
def get_uploaded_image(in_file):
    # Load an uploaded image file
    return Image.open(in_file).convert("RGB")


def resize_image(pil_image):
    # Resize an image to fit within the MAX_WIDTH and MAX_HEIGHT
    if pil_image is None:
        return
    pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)


def display_image_cropped(pil_image, bbox):
    # Display a cropped portion of an image
    cropped_image = pil_image.crop(bbox)
    st.image(cropped_image, use_column_width=True)


@st.cache_data()
def page_count_fn(pdf_file):
    # Return the number of pages in a PDF
    doc = open_pdf(pdf_file)
    return len(doc)


def get_canvas_hash(pil_image):
    return hashlib.md5(pil_image.tobytes()).hexdigest()


@st.cache_data()
def get_image_size(pil_image):
    if pil_image is None:
        return MAX_HEIGHT, MAX_WIDTH
    height, width = pil_image.height, pil_image.width
    return height, width


@st.cache_data(hash_funcs={ImageProcessor: id})
def infer_image(processor, pil_image, bbox):
    # Perform inference on a cropped image
    input_img = pil_image.crop(bbox)
    pred = processor.process_single_image(input_img)

    # Call the function to get the solution from Wolfram Alpha API
    print(pred)
    temp = LatexNodes2Text().latex_to_text(pred)
    text = convert_latex_to_readable(temp)
    print(text)
    solution = asyncio.run(get_solution_from_chatgpt(text))

    return pred, solution


@st.cache_resource()
def load_image_processor(cfg_path):
    processor = ImageProcessor(cfg_path)
    return processor


def convert_latex_to_readable(latex_string):
    # Define a dictionary of LaTeX commands and their replacements
    replacements = {
        r'\\frac{(.*?)}{(.*?)}': r'(\1/\2)',  # Fractions
        r'\\sqrt{(.*?)}': r'sqrt(\1)',  # Square roots
        r'\\sum_{(.*?)}^{(.*?)}': r'sum(\1 to \2)',  # Summations
        r'\\int_{(.*?)}^{(.*?)}': r'int(\1 to \2)',  # Integrals
        r'\\(\\[a-zA-Z]+)': '',  # Remove other LaTeX commands
        r'\^{(.*?)}': r'^(\1)',  # Superscripts
        r'\_\{(.*?)}': r'_(\1)',  # Subscripts
        r'\\alpha': 'α',  # Greek letters
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        # ... add more Greek letters as needed
        r'\\times': '×',  # Multiplication sign
        r'\\cdot': '⋅',  # Dot product
        r'\\div': '÷',  # Division sign
        r'\\pm': '±',  # Plus-minus sign
        r'\\infty': '∞',  # Infinity
        r'\\degree': '°',  # Degree sign
    }

    # Replace LaTeX commands with their replacements
    for pattern, replacement in replacements.items():
        latex_string = re.sub(pattern, replacement, latex_string)
    return latex_string

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.aimlapi.com/v1"
)

async def get_solution_from_chatgpt(expression):
    try:
        response =  await client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": f"Explain, describe and solve the following: {expression}"}],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        print(response)
        solution = response.choices[0].message.content.strip()
        return solution
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Error: Unable to solve the expression."

def run_mode1():
    """Direct Recognition mode: recognize formulas directly from an image
    """
    in_file = st.file_uploader("Input Image:", type=["png", "jpg", "jpeg", "gif", "webp"])

    if in_file is None:
        st.stop()

    filetype = in_file.type
    pil_image = get_uploaded_image(in_file)
    resize_image(pil_image)

    col1, col2 = st.columns([0.5,0.5])

    with col1:
        st.image(pil_image, use_column_width=True)
        st.markdown(
            "<h4 style='text-align: center; color: black;'>[Input: Image] </h4>",
            unsafe_allow_html=True,
        )
        bbox_list = [(0, 0, pil_image.width, pil_image.height)]

    with col2:
        inferences = [infer_image(processor, pil_image, bbox) for bbox in bbox_list]
        for idx, (bbox, (inference, solution)) in enumerate(zip(reversed(bbox_list), reversed(inferences))):
            st.markdown("<h5 style='text-align: center; color: black;'>Prediction: Rendered Expression</h5>", unsafe_allow_html=True)
            st.latex(inference)

        
    inferences = [infer_image(processor, pil_image, bbox) for bbox in bbox_list]
    for idx, (bbox, (inference, solution)) in enumerate(zip(reversed(bbox_list), reversed(inferences))):
        st.markdown("<h3 style='text-align: center; color: black;'>Solution</h3>", unsafe_allow_html=True)
        st.markdown("_____")
        st.markdown(solution)
        st.markdown("_____")

def run_mode2():
    """Manual Selection mode: allows users to select formulas in an image or PDF for recognition.
    """
    
    in_file = st.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])

    if in_file is None:
        st.stop()

    # Determine if the uploaded file is a PDF or an image
    whole_image = False
    if in_file.type == "application/pdf":
        page_count = page_count_fn(in_file)
        page_number = st.number_input("Page number:", min_value=1, value=1, max_value=page_count)
        pil_image = get_page_image(in_file, page_number)
    else:
        pil_image = get_uploaded_image(in_file)
        whole_image = st.button("Formula Recognition")

    resize_image(pil_image)
    canvas_hash = get_canvas_hash(pil_image) if pil_image else "canvas"

    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        # Create a canvas component where users can draw rectangles to select formulas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color="#FFAA00",
            background_color="#FFF",
            background_image=pil_image,
            update_streamlit=True,
            height=get_image_size(pil_image)[0],
            width=get_image_size(pil_image)[1],
            drawing_mode="rect",
            point_display_radius=0,
            key=canvas_hash,
        )

    # Process the drawn rectangles or the whole image if 'whole_image' is True
    if canvas_result.json_data is not None or whole_image:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        bbox_list = []
        if objects.shape[0] > 0:
            boxes = objects[objects["type"] == "rect"][
                ["left", "top", "width", "height"]
            ]
            boxes["right"] = boxes["left"] + boxes["width"]
            boxes["bottom"] = boxes["top"] + boxes["height"]
            bbox_list = boxes[["left", "top", "right", "bottom"]].values.tolist()
        if whole_image:
            bbox_list = [(0, 0, pil_image.width, pil_image.height)]

        if bbox_list:
            with col2:
                inferences = [infer_image(processor, pil_image, bbox) for bbox in bbox_list]
                for idx, (bbox, (inference, solution)) in enumerate(zip(reversed(bbox_list), reversed(inferences))):
                    st.markdown(f"### Result {len(inferences) - idx}")
                    st.markdown("<h6 style='text-align: left; color: black;'>[Input: Image] </h6>", unsafe_allow_html=True)
                    display_image_cropped(pil_image, bbox)
                    st.markdown("<h6 style='text-align: left; color: black;'>[Prediction: Rendered Image] </h6>", unsafe_allow_html=True)
                    st.latex(inference)
                    
            inferences = [infer_image(processor, pil_image, bbox) for bbox in bbox_list]
            for idx, (bbox, (inference, solution)) in enumerate(zip(reversed(bbox_list), reversed(inferences))):
                st.markdown("<h3 style='text-align: center; color: black;'>Solution</h3>", unsafe_allow_html=True)
                st.markdown("_____")
                st.markdown(solution)
                st.markdown("_____")

    with col2:
        tips = """
        ### Usage tips
        - Draw a box around the equation to get the prediction."""
        st.markdown(tips)


if __name__ == "__main__":

    st.set_page_config(layout="wide")
    html_code = """
    <div style='text-align: center; color: black;'>
        <h1>PixSolve</h1>
        <h3>An AI-powered tool for recognizing and solving mathematical and scientific expressions</h3>
    </div>
    """
    readme_text = st.markdown(html_code, unsafe_allow_html=True)
    root_path = os.path.abspath(os.getcwd())
    config_path = os.path.join(root_path, "configs/demo.yaml")
    processor = load_image_processor(config_path)

    app_mode = st.selectbox(
        "Switch Mode:", ["Direct Recognition", "Manual Selection"]
    )

    # Direct Recognition: Input an image containing formulas and output the recognition results.
    if app_mode == "Direct Recognition":
        st.markdown("---")
        st.markdown(
            f"<h3 style='text-align: center; color: red;'> Direct Recognition </h3>"
            f"<br>"
            f"<h5 style='text-align: center; color: black;'>Input an image containing formulas and output the solution.</h5>",
            unsafe_allow_html=True,
        )
        run_mode1()

    # Manual Selection: Input a document or webpage screenshot, detect all formulas, then recognize each one.
    elif app_mode == "Manual Selection":
        st.markdown("---")
        st.markdown(
            f"<h3 style='text-align: center; color: red;'> Manual Selection and Recognition </h3>"
            f"<br>"
            f"<h5 style='text-align: center; color: black;'>Input a document or webpage screenshot, detect all formulas, then recognize each one.</h5>",
            unsafe_allow_html=True,
        )
        run_mode2()
