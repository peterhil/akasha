#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Unit tests for oscillator.py
"""

from oscillator import *
from fractions import Fraction
import unittest

class TuningTest(unittest.TestCase):
    """Tests for asserting that Osc has access to sample rate."""
    def testTuning(self):
        self.failUnlessRaises(AttributeError, Osc.tuning)
        Osc.tuning = 44100
        self.assertEqual(44100, Osc.tuning)
    
class OscInitTest(unittest.TestCase):
    def setUp(self):
        self.o8 = Osc(Fraction(1,8))
        self.p8 = Osc(Fraction(9,8))
        self.m8 = Osc(Fraction(-1,8))
        
    def testInit(self):
        """Test initialization"""
        assert(isinstance(self.o8, Osc))
        
    def testModularWrap(self):
        self.assertEqual(self.o8, self.p8)
        self.assertEqual(7, self.m8.order)
        
    def testSize(self):
        """Test size of roots"""
        self.assertEqual(8, self.o8.roots.size)
    

if __name__ == "__main__":
    unittest.main()