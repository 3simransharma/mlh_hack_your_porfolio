"use client";

import React, { useState } from "react";
import * as tf from '@tensorflow/tfjs';

export default function Home() {
  const [ingredient, setIngredient] = useState('');
  const [recipe, setRecipe] = useState('');
  const [substitute, setSubstitute] = useState('');

  const handleClick = async () => {
    try {
      // Load the model
      const model = await tf.loadLayersModel('/models/model/model.json');

      // Prepare the input data
      const vocab = [
        "flour", "butter", "sugar", "milk", "egg", "yogurt", "cream", "rice",
        "maida", "margarine", "honey", "almond milk", "flaxseed meal", "coconut yogurt",
        "soy cream", "quinoa", "tofu", "tempeh", "ghee", "paneer", "curd", "basmati rice",
        "jaggery", "mustard oil", "chana dal", "coconut", "masoor dal", "turmeric",
        "green chili", "urad dal", "tamarind", "besan", "amchur", "pistachio", "mango",
        "kesar", "ajwain", "coriander", "ginger", "garlic", "onion", "tomato", "cumin",
        "mustard seeds", "fennel seeds", "cardamom", "cinnamon", "cloves", "fenugreek",
        "asafoetida", "bay leaves", "curry leaves", "masala", "olive oil", "garlic powder",
        "onion powder", "baking soda", "baking powder", "vanilla extract", "chocolate chips",
        "brown sugar", "maple syrup", "oats", "peanut butter", "strawberries", "blueberries",
        "spinach", "kale", "avocado", "lettuce", "carrot", "broccoli", "cauliflower",
        "zucchini", "bell pepper", "mushroom", "bacon", "sausage", "ham", "steak",
        "ground beef", "salmon", "shrimp", "tuna", "cod", "crab", "lobster", "scallops",
        "pasta", "spaghetti", "macaroni", "lasagna", "risotto", "potatoes", "fries",
        "cheese", "cream cheese", "sour cream", "whipped cream", "gelatin", "bread crumbs",
        "breadcrumbs", "panko", "soy sauce", "mirin", "sake", "miso", "dashi", "wasabi",
        "nori", "katsuobushi", "furikake", "sesame seeds", "matcha", "red bean paste",
        "shoyu", "tofu", "edamame", "shiitake", "enoki", "daikon", "burdock", "lotus root",
        "taro", "shiso", "yuzu", "umeboshi", "kewpie mayonnaise", "bulgogi", "kimchi",
        "gochujang", "tamarind paste", "fish sauce", "oyster sauce", "hoisin sauce",
        "sesame oil", "palm sugar", "rice vinegar", "soba noodles", "udon noodles",
        "ramen noodles", "tempura batter", "panko bread crumbs", "yuzu kosho", "yuba",
        "maltose", "maltodextrin", "agar agar", "konnyaku", "natto", "sukiyaki",
        "mochi", "anko", "kinako", "takuan", "hakusai", "mibuna", "mizuna", "kabu",
        "kamaboko", "aburaage", "menma", "takoyaki sauce", "okonomiyaki sauce", "yakiniku sauce",
        "ponzu", "shichimi", "nanami", "karashi", "kanzuri", "shirako", "milt", "katsu",
        "menchi", "mentaiko", "ikura", "masago", "tobiko", "unagi", "anago", "kabayaki",
        "sunomono", "nimono", "agemono", "nabemono", "yakimono", "tsukemono", "kobujime",
        "kaeshi", "mitarashi", "kakigori", "yokan", "dorayaki", "taiyaki", "imagawayaki"
      ];

      const tokenizeText = (text, vocab) => {
        return text.split(' ').map(word => vocab.indexOf(word) + 1 || 0);
      };

      const tokenizedInput = tokenizeText(`${ingredient} ${recipe}`, vocab);
      const maxLen = 40;  // Updated to match the expected input shape of the model
      const paddedInput = new Array(maxLen).fill(0);
      tokenizedInput.forEach((val, idx) => {
        paddedInput[idx] = val;
      });

      const inputTensor = tf.tensor2d([paddedInput], [1, maxLen]);

      // Make prediction
      const prediction = model.predict(inputTensor);
      const predictedIndex = prediction.argMax(-1).dataSync()[0];
      const predictedSubstitute = vocab[predictedIndex - 1];

      setSubstitute(predictedSubstitute);
    } catch (error) {
      console.error("Error making prediction:", error);
      setSubstitute("Error finding substitute");
    }
  };

  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen font-bold" style={{ backgroundColor: '#FFF2E5' }}>
      <div className="absolute inset-0 z-0" style={{ background: "url('/squaredBg.svg') no-repeat center center", backgroundSize: 'cover' }}></div>


      <div className="absolute top-4 left-4 z-20 p-3 ml-5">
        <h1 className="text-4xl" style={{ color: '#803219' }}>Yammi!</h1>
      </div>

      <div>
        <img src="/elements/1.svg" alt="Image 4" style={{ top:"100px", width: '150px', height: 'auto', left: '20px', position: 'absolute' }} />
        <img src="/elements/2.svg" alt="Image 5" style={{ top:"100px", left: "150px", width: '100px', height: 'auto', position: 'absolute' }} />
      </div>

      

      <div className="flex flex-col items-center justify-center z-10">
        <h2 className="text-7xl text-center mb-8" style={{ color: '#803219' }}>Substitute ingredients, not flavor!</h2>
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            placeholder="Substitute What?"
            value={ingredient}
            onChange={(e) => setIngredient(e.target.value)}
            className="p-4 rounded-2xl text-lg w-64 input"
            style={{ backgroundColor: '#FFDDBA', color: '#803219' }}
          />
          <input
            type="text"
            placeholder="For which recipe?"
            value={recipe}
            onChange={(e) => setRecipe(e.target.value)}
            className="p-4 rounded-2xl text-lg w-64 input"
            style={{ backgroundColor: '#FFDDBA', color: '#803219' }}
          />
        </div>

        <button className="px-6 py-3 rounded-xl text-white" style={{ backgroundColor: '#803219' }} onClick={handleClick}>
          Find Substitute
        </button>
        <div>
      <img src="/elements/3.svg" alt="Image 6" style={{ top:"150px", right: "200px", width: '100px', height: 'auto', position: 'absolute' }} />
      </div>


        {substitute && (
          <div className="mt-12 p-4 rounded-lg answer" >
            Substitute ingredient of {ingredient} is: {substitute}
          </div>
        )}



        <div style={{ position: 'absolute', bottom: '50px', right: '4px', zIndex: '20', display: 'flex', flexDirection: 'row', alignItems: 'flex-end' }}>
          <img src="/elements/4.svg" alt="Image 1" style={{ width: '150px', height: 'auto', marginLeft: '-20px', top:"200", left:'200px',transform: 'rotate(-30deg)' }} />
          <img src="/elements/5.svg" alt="Image 2" style={{ width: '180px', height: 'auto', marginRight: '40px' }} />
          <img src="/elements/6.svg" alt="Image 3" style={{ width: '150px', left:'100px',height: 'auto', marginLeft: '-10px' }} />
          {/* <img src="/elements/7.svg" alt="Image 4" style={{ width: '100px', height: 'auto' }} /> */}
          {/* <img src="/elements/8.svg" alt="Image 5" style={{ width: '100px', height: 'auto' }} /> */}
        </div>
      </div>

      <footer className="absolute bottom-4 text-lg z-10" style={{ color: '#803219' }}>
        Made with ❤️ from Simran!
      </footer>
    </div>
  );
}
