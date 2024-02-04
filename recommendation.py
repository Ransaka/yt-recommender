from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from models.encoder import Encoder
from indexer import Indexer
import torch
import os


model = SentenceTransformer('intfloat/multilingual-e5-base')

encoder = Encoder()
encoder.load_state_dict(torch.load('./models/encoder.bin', map_location=torch.device('cpu')))

dataset = load_dataset("Ransaka/youtube_recommendation_data", token=os.environ.get('HF'))
dataset = concatenate_datasets([dataset['train'], dataset['test']])

latent_data = torch.load("data/latent_data_final.bin")
embeddings = torch.load("data/embeddings.bin")

text_embedding_index = Indexer(embeddings)
image_embedding_index = Indexer(latent_data)

def get_recommendations(image, title, k):
#   title = [dataset[product_id]['title']]
  title_embeds = model.encode([title], normalize_embeddings=True)
  image = transforms.ToTensor()(image.convert("L"))
  image_embeds =  encoder(image).detach().numpy()

  image_candidates = image_embedding_index.topk(image_embeds,k=k)
  title_candidates = text_embedding_index.topk(title_embeds, k=k)
  final_candidates = []
  final_candidates.append(list(image_candidates[0]))
  final_candidates.append(list(title_candidates[0]))
  final_candidates = sum(final_candidates,[])
  final_candidates = list(set(final_candidates))
  results_dict = {"image":[], "title":[]}
  for candidate in final_candidates:
    results_dict['image'].append(dataset['image'][candidate])
    results_dict['title'].append(dataset['title'][candidate])
  return results_dict