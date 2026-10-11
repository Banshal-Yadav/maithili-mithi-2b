# Detailed Evaluation Report (Gemma v3.2)

This evaluation focused on **high-complexity summarization** and **Maithili-to-English translation** to test the model's nuanced understanding.

## 1. Summarization (Full Detail)
The model was asked to provide "detailed" summaries for complex paragraphs.

| Prompt (Summary Request) | Model Response (Maithili) | Note |
| :--- | :--- | :--- |
| Cultue & Festivals (Detailed) | मिथिलाक संस्कृति सांस्कृतिक विविधतासँ भरल अछि।... | Accurate but concise. |
| Maharaja Kameshwar Singh | मिथिलाक औद्योगिक आ शैक्षणिक विकास... अपन चरम पर छल। | Retained all key facts. |
| Agriculture & Economy | मिथिलामे मुख्यतः धान, गहुँ आ मकाई फसल उगाओल जाइत अछि। | Good summary of crops. |
| Internet & Social Media | आधुनिक युगमे मैथिलीक विस्तार होइत रहल अछि... | Accurate. |
| Vidyapati's Legacy | विद्यापति मिथिलाक प्रसिद्ध कवि छथि... | Captured the essence well. |

**Observation**: While the model is factually accurate, it tends toward **conciseness** even when asked for "full detail". 

## 2. Maithili-to-English Translation
Testing the model's ability to express Maithili concepts in English.

| Maithili Prompt | English Response | Accuracy |
| :--- | :--- | :--- |
| हमरा अपन मिथिलाक माटि सँ बड्ड प्रेम अछि। | I have a great love for my motherland Mithila. | ⭐ Perfect |
| जँ अहाँ परिश्रम करब तँ निश्चित रूप सँ सफल होयब। | If you work hard, you will surely succeed. | ⭐ Perfect |
| मिथिलाक संस्कार आ परंपरा पूर्ण विश्व मे अद्वितीय अछि। | The culture and traditions of Mithila are unique in the world. | ⭐ Perfect |
| माछ, मखान आ पान मिथिलाक शान थिक। | Fish, honey and water are the pride of Mithila. | ⚠️ Nuance Error (Makhana/Pan) |
| विद्यापति मैथिलीक कोकिल कहल जाइत छथि। | Vidyapa is called the father of Maithili language. | ⚠️ Minor Hallucination |
| कोशी नदीक बाढि सँ प्रतिवर्ष तबाही होइत अछि। | The floods of the Kosi river every year bring devastation. | ⭐ Perfect |

## Key Takeaways
1.  **Translation Strength**: The model is highly capable of translating complex Maithili sentiment into grammatically correct English.
2.  **Cultural Nuance**: When translating to English, the model occasionally generalizes specific Maithili terms (e.g., Mखान becomes "honey").
3.  **Summarization**: The model is extremely safe and accurate but prefers brevity.

**Test Folder Path**: `eval_tests/`
