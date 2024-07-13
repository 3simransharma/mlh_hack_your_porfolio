import * as tf from '@tensorflow/tfjs';
import { promises as fs } from 'fs';
import path from 'path';

let model;
let substitutesData;
let vocab;

async function loadModel() {
  if (!model) {
    model = await tf.loadLayersModel('file://./models/model/model.json');
  }
}

async function loadSubstitutesData() {
  if (!substitutesData) {
    const dataPath = path.join(process.cwd(), 'public', 'data', 'dataset.json');
    const jsonData = await fs.readFile(dataPath, 'utf8');
    substitutesData = JSON.parse(jsonData);
    vocab = createVocab(substitutesData);
  }
}

const tokenizeText = (text, vocab) => {
  return text.split(' ').map(word => vocab.indexOf(word) + 1 || 0);
};

const createVocab = (data) => {
  const vocab = [];
  data.forEach(item => {
    item.ingredient.split(' ').forEach(word => {
      if (!vocab.includes(word)) vocab.push(word);
    });
    item.recipe.split(' ').forEach(word => {
      if (!vocab.includes(word)) vocab.push(word);
    });
    item.substitute.split(' ').forEach(word => {
      if (!vocab.includes(word)) vocab.push(word);
    });
  });
  return vocab;
};

export async function POST(request) {
  try {
    await loadModel();
    await loadSubstitutesData();
    const { ingredient, recipe } = await request.json();

    if (!ingredient || !recipe) {
      return new Response(JSON.stringify({ message: 'Ingredient and recipe are required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const tokenizedInput = tokenizeText(ingredient + ' ' + recipe, vocab);
    const paddedInput = tf.tensor2d([tokenizedInput], [1, tokenizedInput.length]);

    const prediction = model.predict(paddedInput);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];

    const substitute = vocab[predictedIndex - 1];

    return new Response(JSON.stringify({ substitute }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error parsing request body:', error);
    return new Response(JSON.stringify({ message: 'Invalid request' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
