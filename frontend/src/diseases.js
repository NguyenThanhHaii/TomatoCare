export const CLASS_NAMES = ['Healthy', 'Early Blight', 'Late Blight', 'Leaf Mold'];

export const DISEASES = {
  Healthy: {
    title: 'Healthy leaf',
    description: 'The model finds this photo most similar to its Healthy class. A photo alone cannot rule out disease or problems outside the four supported classes.',
    recommendations: ['Continue regular leaf inspections.', 'Water near the soil and keep foliage dry.', 'Maintain good spacing and air circulation.', 'Ask a local plant specialist if symptoms develop.'],
  },
  'Early Blight': {
    title: 'Early Blight',
    description: 'Early blight can cause brown spots with concentric rings, often on older lower leaves. Other problems can look similar; confirm symptoms with a local plant specialist.',
    recommendations: ['Consider removing visibly affected leaves.', 'Improve air circulation around plants.', 'Avoid overhead watering and wet foliage.', 'Ask a local specialist about suitable treatment if symptoms spread.'],
  },
  'Late Blight': {
    title: 'Late Blight',
    description: 'Late blight may cause irregular, water-soaked dark patches and spread quickly in cool, humid conditions. Prompt professional advice is useful when it is suspected.',
    recommendations: ['Inspect nearby plants for spreading symptoms.', 'Keep foliage dry and improve ventilation.', 'Separate affected plants where practical.', 'Seek local plant-health advice promptly if patches spread.'],
  },
  'Leaf Mold': {
    title: 'Leaf Mold',
    description: 'Leaf mold is associated with humid conditions, yellow upper-leaf patches and olive-brown growth underneath. Inspect both sides of leaves and seek confirmation.',
    recommendations: ['Reduce humidity and improve ventilation.', 'Avoid wetting leaves when watering.', 'Consider removing heavily affected leaves.', 'Consult a local specialist for persistent symptoms.'],
  },
};
