import unittest
import os
import sys
import types

# Stub heavy dependencies before importing modules under test
sys.modules.setdefault('pyaudio', types.SimpleNamespace(paInt16=8, PyAudio=lambda: None))
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('librosa', types.ModuleType('librosa'))
sys.modules.setdefault('transformers', types.ModuleType('transformers'))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.text_to_speech import Speech
from sources.speech_to_text import Transcript

class TestSpeechUtils(unittest.TestCase):
    def test_shorten_paragraph(self):
        sp = Speech(enable=False)
        text = "**Explanation**: This is the first sentence. This is the second one."
        result = sp.shorten_paragraph(text)
        self.assertEqual(result, "**Explanation**: This is the first sentence.")

    def test_remove_hallucinations(self):
        trans = Transcript.__new__(Transcript)  # bypass heavy initialization
        text = "Okay. Thank you for watching. You're going to love it."
        result = trans.remove_hallucinations(text)
        self.assertEqual(result, "love it.")

if __name__ == '__main__':
    unittest.main()
