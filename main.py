import os
import requests
import replicate
from PIL import Image
from io import BytesIO
import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import Replicate as LangChainReplicate
from langchain_core.prompts import PromptTemplate

# Streamlit app
st.title("Image Generation with Replicate API")

# Input box for the Replicate API token
api_token = st.text_input("Enter your Replicate API Token", type="password")

if api_token:
    # Set the Replicate API token
    os.environ["REPLICATE_API_TOKEN"] = api_token
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    # Input fields for the parameters
    cfg = st.number_input("CFG (Classifier-Free Guidance)", value=4)
    image_url = st.text_input("Image URL", value="https://replicate.delivery/pbxt/LI5VAhU2v3jNTjuE76GMTzikT1XMiUoRSznZdXR0cAnK1XJS/ComfyUI_00362_.png")
    steps = st.number_input("Steps", value=25)
    width = st.number_input("Width", value=1024)
    height = st.number_input("Height", value=1024)
    prompt_input = st.text_input("Enter a prompt idea", value="an illustration of a cute dog jumping over a sleeping cat")
    sampler = st.selectbox("Sampler", ["dpmpp_2m_sde_gpu", "other_sampler_options"])
    scheduler = st.selectbox("Scheduler", ["karras", "other_scheduler_options"])
    output_format = st.selectbox("Output Format", ["webp", "other_formats"])
    output_quality = st.number_input("Output Quality", value=80)
    negative_prompt = st.text_input("Negative Prompt", value="")
    number_of_images = st.number_input("Number of Images", value=1)
    ip_adapter_weight = st.number_input("IP Adapter Weight", value=1.0)
    ip_adapter_weight_type = st.selectbox("IP Adapter Weight Type", ["style transfer precise", "other_weight_types"])

    if st.button("Generate Image"):
        # Use LangChain to generate the prompt
        template = """
        Create a detailed and creative image description based on the following idea:
        {prompt_input}
        """
        prompt = PromptTemplate(template=template, input_variables=["prompt_input"])
        llm = LangChainReplicate(
            model="meta/meta-llama-3-8b-instruct",
            model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        generated_prompt = llm_chain.run({"prompt_input": prompt_input})

        # Run the replicate model
        output = replicate.run(
            "fofr/kolors-with-ipadapter:5a1a92b2c0f81813225d48ed8e411813da41aa84e7582fb705d1af46eea36eed",
            input={
                "cfg": cfg,
                "image": image_url,
                "steps": steps,
                "width": width,
                "height": height,
                "prompt": generated_prompt,
                "sampler": sampler,
                "scheduler": scheduler,
                "output_format": output_format,
                "output_quality": output_quality,
                "negative_prompt": negative_prompt,
                "number_of_images": number_of_images,
                "ip_adapter_weight": ip_adapter_weight,
                "ip_adapter_weight_type": ip_adapter_weight_type
            }
        )

        st.write("Generated Image:")
        for img_url in output:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Generated Image")
