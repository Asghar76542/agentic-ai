import unittest
import os
import sys
import types

sys.modules.setdefault('pyaudio', types.SimpleNamespace(paInt16=0))
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('librosa', types.ModuleType('librosa'))
sys.modules.setdefault('transformers', types.ModuleType('transformers'))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.text_to_speech import Speech
from sources.speech_to_text import Transcript

class TestAudioUtils(unittest.TestCase):
    def test_shorten_paragraph(self):
        tts = Speech(enable=False)
        text = "**Explanation**: This is the first sentence. This is the second sentence.\nNext line"
        result = tts.shorten_paragraph(text)
        self.assertEqual(result, "**Explanation**: This is the first sentence.\nNext line")

    def test_remove_hallucinations(self):
        tr = Transcript.__new__(Transcript)
        sample = "Okay. Thank you for watching. This is valid text."
        cleaned = tr.remove_hallucinations(sample)
        self.assertEqual(cleaned, "This is valid text.")

if __name__ == '__main__':
    unittest.main()
