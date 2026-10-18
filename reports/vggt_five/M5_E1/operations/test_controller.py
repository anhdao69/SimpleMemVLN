import unittest
from controller import execute_sequence, ready_to_qualify

class QueueTests(unittest.TestCase):
 def exercise(self, fail=None, complete=True):
  events=[]
  def stage(name):
   def run():
    events.append(name)
    if fail==name:raise RuntimeError(name)
   return run
  return events,lambda:execute_sequence(stage('qualify'),stage('triple'),lambda:complete,stage('m5'))
 def test_order(self):
  e,f=self.exercise();f();self.assertEqual(e,['qualify','triple','m5'])
 def test_failed_qualification_still_resumes_triple(self):
  e,f=self.exercise('qualify')
  with self.assertRaises(RuntimeError):f()
  self.assertEqual(e,['qualify','triple'])
 def test_failed_triple_blocks_m5(self):
  e,f=self.exercise('triple')
  with self.assertRaises(RuntimeError):f()
  self.assertEqual(e,['qualify','triple'])
 def test_zero_exit_incomplete_blocks_m5(self):
  e,f=self.exercise(complete=False)
  with self.assertRaises(RuntimeError):f()
  self.assertEqual(e,['qualify','triple'])
 def test_active_compute_blocks_qualification(self):
  self.assertFalse(ready_to_qualify(['4925.0','4925.124','4925.125'],'4925.125','4925'))
  self.assertTrue(ready_to_qualify(['4925.0','4925.125'],'4925.125','4925'))
  self.assertFalse(ready_to_qualify(['4925.0','4925.125','4925.999'],'4925.125','4925'))
if __name__=='__main__':unittest.main()
