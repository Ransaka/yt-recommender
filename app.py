import streamlit as st
import random
import os
from datasets import load_dataset, concatenate_datasets
from recommendation import get_recommendations


START = random.randint(a=1,b=200)
END = START + 10

dataset = load_dataset("Ransaka/youtube_recommendation_data", token=os.environ.get('HF'))
dataset = concatenate_datasets([dataset['train'], dataset['test']])



pil_images = dataset['image'][START:END]

def show_image_metadata_and_related_info(image_index):
    selected_image = pil_images[image_index]
    image_title = dataset['title'][image_index]
    st.image(selected_image, caption=f"{image_title}", use_column_width=True)
    
    with st.expander("You May Also Like.."):
        dataset_s = get_recommendations(selected_image,image_title,8)
        
        col1_row1, col2_row1 = st.columns(2)
        with col1_row1:
            st.image(image=dataset_s['image'][0], caption=dataset_s['title'][0], width=200)
        with col2_row1:
            st.image(image=dataset_s['image'][1], caption=dataset_s['title'][1], width=200)

        # Second Row
        col1_row2, col2_row2 = st.columns(2)
        with col1_row2:
            st.image(image=dataset_s['image'][2], caption=dataset_s['title'][2], width=200)
        with col2_row2:
            st.image(image=dataset_s['image'][3], caption=dataset_s['title'][3], width=200)

        # Third row
        col1_row3, col2_row3 = st.columns(2)
        with col1_row3:
            st.image(image=dataset_s['image'][4], caption=dataset_s['title'][4], width=200)
        with col2_row3:
            st.image(image=dataset_s['image'][5], caption=dataset_s['title'][5], width=200)

        # Fourth Row
        col1_row4, col2_row4 = st.columns(2)
        with col1_row4:
            st.image(image=dataset_s['image'][6], caption=dataset_s['title'][6], width=200)
        with col2_row4:
            st.image(image=dataset_s['image'][7], caption=dataset_s['title'][7], width=200)

def main():
    st.title("Youtube Recommendation Engine")

    for i, image in enumerate(pil_images):
        show_image_metadata_and_related_info(i)

if __name__ == '__main__':
    main()