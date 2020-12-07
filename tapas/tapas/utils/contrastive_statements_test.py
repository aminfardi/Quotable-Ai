# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3

import random
import re
from typing import Text, List, Mapping

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tapas.protos import annotated_text_pb2
from tapas.protos import interaction_pb2
from tapas.utils import contrastive_statements

FLAGS = flags.FLAGS

ContrastiveCandidate = contrastive_statements.ContrastiveCandidate
RewriteResult = contrastive_statements.RewriteResult


def create_interaction(
    table,
    statement,
    mentions,
):
  interaction = interaction_pb2.Interaction()
  for index, row in enumerate(table):
    new_row = interaction.table.columns
    if index > 0:
      new_row = interaction.table.rows.add().cells
    for cell in row:
      new_row.add().text = cell
  for row in interaction.table.rows:
    for cell in row.cells:
      cell_text = cell.text.lower()
      if cell_text not in mentions:
        continue
      annotated_text = cell.Extensions[\
          annotated_text_pb2.AnnotatedText.annotated_cell_ext]
      annotation = annotated_text.annotations.add()
      annotation.begin_byte_index = 0
      annotation.end_byte_index = len(cell_text)
      annotation.identifier = mentions[cell_text]

  question = interaction.questions.add()
  question.original_text = statement
  q_annotated_text = question.Extensions[\
    annotated_text_pb2.AnnotatedText.annotated_question_ext]
  question_text = question.original_text.lower()
  for phrase, identifier in mentions.items():
    for match in re.finditer(phrase, question_text):
      annotation = q_annotated_text.annotations.add()
      begin, end = match.span()
      annotation.begin_byte_index = begin
      annotation.end_byte_index = end
      annotation.identifier = identifier
  return interaction


class ContrastiveStatementsTest(parameterized.TestCase):

  def get_interaction(self):
    interaction = create_interaction(
        [
            ['Name', 'Age', 'Birthday'],
            ['Bob', '1.7', '24 April 1950'],
            ['Julia', '1.5', '24 April 1951'],
            ['Peter', '1.9', '24 March 1950'],
        ], 'Robert was born on 24 April 1950.', {
            'bob': 'http://en.wikipedia.org/wiki/Bob',
            'robert': 'http://en.wikipedia.org/wiki/Bob',
            'peter': 'http://en.wikipedia.org/wiki/Peter',
            'julia': 'http://en.wikipedia.org/wiki/Julia',
        })
    return interaction

  def test_candidates(self):
    interaction = self.get_interaction()

    candidates = list(
        contrastive_statements.get_contrastive_candidates(interaction))

    # Make the order canonical for list comparison.
    candidates.sort(key=lambda cand: cand.focus_texts)

    logging.info(candidates)

    self.assertEqual(candidates, [
        ContrastiveCandidate(
            text='Robert was born on 24 April 1950.',
            focus_texts={'24 April 1950'},
            positive_replacement_texts={'24 April 1950'},
            negative_replacement_texts={'24 March 1950', '24 April 1951'},
            support_texts={'Robert'}),
        ContrastiveCandidate(
            text='Robert was born on 24 April 1950.',
            focus_texts={'Bob', 'Robert'},
            positive_replacement_texts={'Bob'},
            negative_replacement_texts={'Julia', 'Peter'},
            support_texts={'24 April 1950'})
    ])

  def test_candidates_remove_ambiguous_entities(self):
    # Same example as above but all references point to Bob.
    interaction = create_interaction(
        [
            ['Name', 'Age', 'Birthday'],
            ['Bob', '1.7', '24 April 1950'],
            ['Julia', '1.5', '24 April 1951'],
            ['Peter', '1.9', '24 March 1950'],
        ], 'Bob was born on 24 April 1950.', {
            'bob': 'http://en.wikipedia.org/wiki/Bob',
            'peter': 'http://en.wikipedia.org/wiki/Bob',
            'julia': 'http://en.wikipedia.org/wiki/Bob',
        })

    candidates = list(
        contrastive_statements.get_contrastive_candidates(interaction))

    # Make the order canonical for list comparison.
    candidates.sort(key=lambda cand: cand.focus_texts)

    logging.info(candidates)

    self.assertEqual(candidates, [
        ContrastiveCandidate(
            text='Bob was born on 24 April 1950.',
            focus_texts={'24 April 1950'},
            positive_replacement_texts={'24 April 1950'},
            negative_replacement_texts={'24 March 1950', '24 April 1951'},
            support_texts={'Bob'}),
    ])

  def test_candidates_honor_reference_types(self):
    """Makes sure we don't replace ordinal with cardinals and vice versa."""
    interaction = create_interaction([
        ['Name', 'Age', 'Rank'],
        ['Bob', '1.7', '5th'],
        ['Julia', '1.5', '2'],
        ['Peter', '1.9', 'third'],
    ], 'Bob is 5th.', {})

    candidates = list(
        contrastive_statements.get_contrastive_candidates(interaction))

    # Make the order canonical for list comparison.
    candidates.sort(key=lambda cand: cand.focus_texts)

    logging.info(candidates)

    self.assertEqual(candidates, [
        ContrastiveCandidate(
            text='Bob is 5th.',
            focus_texts={'5th'},
            positive_replacement_texts={'5th'},
            negative_replacement_texts={'third'},
            support_texts=set()),
    ])

  def test_get_random_contrastive_statement_simple(self):
    rewrite = contrastive_statements.Rewriter()
    candidate = ContrastiveCandidate(
        text='Robert was born on 24 April 1950.',
        focus_texts={'24 April 1950'},
        positive_replacement_texts={'24 April 1950'},
        negative_replacement_texts={'24 March 1950', '24 April 1951'},
        support_texts={'Robert'})
    texts = set()
    for i in range(10):
      rng = random.Random(i)
      texts.add(
          rewrite.get_random_contrastive_statement(rng, candidate,
                                                   candidate.text))
    self.assertEqual(
        texts, {
            RewriteResult(True, 'Robert was born on 24 April 1951.', 7),
            RewriteResult(True, 'Robert was born on 24 March 1950.', 7),
        })

  def test_get_random_contrastive_statement_no_focus(self):
    rewrite = contrastive_statements.Rewriter()
    candidate = ContrastiveCandidate(
        text='Robert was born on 24 May 1950.',
        focus_texts={'24 April 1950'},
        positive_replacement_texts={'24 April 1950'},
        negative_replacement_texts={'24 March 1950', '24 April 1951'},
        support_texts={'Robert'})
    texts = set()
    for i in range(10):
      rng = random.Random(i)
      texts.add(
          rewrite.get_random_contrastive_statement(rng, candidate,
                                                   candidate.text))
    self.assertEqual(texts, {None})

  def test_get_random_contrastive_statement_no_support(self):
    rewrite = contrastive_statements.Rewriter()
    candidate = ContrastiveCandidate(
        text='Peter was born on 24 April 1950.',
        focus_texts={'24 April 1950'},
        positive_replacement_texts={'24 April 1950'},
        negative_replacement_texts={'24 March 1950'},
        support_texts={'Robert'})
    texts = set()
    for i in range(10):
      rng = random.Random(i)
      texts.add(
          rewrite.get_random_contrastive_statement(rng, candidate,
                                                   candidate.text))
    self.assertEqual(
        texts, {RewriteResult(False, 'Peter was born on 24 March 1950.', 7)})

  def test_get_random_contrastive_word_boundaries(self):
    rewrite = contrastive_statements.Rewriter()
    candidate = ContrastiveCandidate(
        text='Peter was born on 24 April 1950.',
        focus_texts={'95', '24 Ap', '4 April'},
        positive_replacement_texts={'24 April 1950'},
        negative_replacement_texts={'25 April 1950'},
        support_texts={'Peter'})
    texts = set()
    for i in range(10):
      rng = random.Random(i)
      texts.add(
          rewrite.get_random_contrastive_statement(rng, candidate,
                                                   candidate.text))
    self.assertEqual(texts, {None})

  def test_end2end(self):
    """Tests contrastive logic end to end.

    Also test that the positive statemnt is 'Bob ...' instead of 'Robert ...'.
    Since we make sure to use the mention in the table instead of the one in the
    text.
    """
    interaction = self.get_interaction()
    pairs = set()
    for i in range(10):
      rng = random.Random(i)
      pairs.update(
          contrastive_statements.get_contrastive_statements(rng, interaction))
    logging.info(pairs)
    self.assertEqual(
        pairs, {
            (True, 'Bob was born on 24 April 1950.',
             'Julia was born on 24 April 1950.'),
            (True, 'Robert was born on 24 April 1950.',
             'Robert was born on 24 April 1951.'),
            (True, 'Robert was born on 24 April 1950.',
             'Robert was born on 24 March 1950.'),
            (True, 'Bob was born on 24 April 1950.',
             'Peter was born on 24 April 1950.'),
        })

  def test_end2end_filter_when_few_context_tokens(self):
    interaction = self.get_interaction()
    pairs = set()
    for i in range(10):
      rng = random.Random(i)
      pairs.update(
          contrastive_statements.get_contrastive_statements(
              rng,
              interaction,
              min_num_context_tokens=8,
          ))
    logging.info(pairs)
    self.assertEmpty(pairs)


if __name__ == '__main__':
  absltest.main()
