"""
Unit tests for the process utility module.
"""

import pytest
from conversion_subnet.utils.process import group_and_merge_boxes


class TestGroupAndMergeBoxes:
    """Test suite for group_and_merge_boxes function."""
    
    def test_empty_input(self):
        """Test with empty input list."""
        result = group_and_merge_boxes([])
        assert result == []

    def test_single_box(self):
        """Test with single box."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Hello'}
        ]
        result = group_and_merge_boxes(data)
        assert len(result) == 1
        assert result[0]['text'] == 'Hello'
        assert result[0]['position'] == [10, 20, 50, 30]

    def test_none_values_filtered(self):
        """Test that None values are filtered out."""
        data = [
            None,
            {'position': [10, 20, 50, 30], 'text': 'Hello'},
            None,
            {'position': [60, 20, 100, 30], 'text': 'World'}
        ]
        result = group_and_merge_boxes(data)
        
        # The two boxes are close enough horizontally to be merged (10 pixels apart, within default xtol=25)
        assert len(result) == 1  # They should merge into one box
        assert result[0]['text'] == 'Hello World'
        assert all('text' in box for box in result)

    def test_invalid_boxes_filtered(self):
        """Test that boxes without 'position' key are filtered out."""
        data = [
            {'text': 'Invalid box'},  # No position key
            {'position': [10, 20, 50, 30], 'text': 'Valid box'},
            {'other_key': 'value'},  # No position key
        ]
        result = group_and_merge_boxes(data)
        assert len(result) == 1
        assert result[0]['text'] == 'Valid box'

    def test_horizontal_merge_within_tolerance(self):
        """Test merging boxes that are horizontally close within tolerance."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Hello'},
            {'position': [55, 20, 95, 30], 'text': 'World'}  # 5 pixels apart (within default xtol=25)
        ]
        result = group_and_merge_boxes(data)
        assert len(result) == 1
        assert result[0]['text'] == 'Hello World'
        assert result[0]['position'] == [10, 20, 95, 30]  # Merged bounding box

    def test_horizontal_no_merge_outside_tolerance(self):
        """Test that boxes outside horizontal tolerance are not merged."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Hello'},
            {'position': [80, 20, 120, 30], 'text': 'World'}  # 30 pixels apart (outside default xtol=25)
        ]
        result = group_and_merge_boxes(data)
        assert len(result) == 2
        assert result[0]['text'] == 'Hello'
        assert result[1]['text'] == 'World'

    def test_vertical_grouping_within_tolerance(self):
        """Test that boxes are grouped into lines when vertically close."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Line1_Part1'},
            {'position': [60, 22, 100, 32], 'text': 'Line1_Part2'},  # 2 pixels apart vertically (within ytol=5)
            {'position': [10, 50, 50, 60], 'text': 'Line2_Part1'},
            {'position': [60, 52, 100, 62], 'text': 'Line2_Part2'}   # 2 pixels apart vertically
        ]
        result = group_and_merge_boxes(data)
        
        # Should have 2 merged boxes (one per line)
        assert len(result) == 2
        
        # Check that text was merged within lines
        texts = [box['text'] for box in result]
        assert 'Line1_Part1 Line1_Part2' in texts
        assert 'Line2_Part1 Line2_Part2' in texts

    def test_vertical_no_grouping_outside_tolerance(self):
        """Test that boxes outside vertical tolerance are not grouped."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Line1'},
            {'position': [60, 40, 100, 50], 'text': 'Line2'}  # 10 pixels apart vertically (outside ytol=5)
        ]
        result = group_and_merge_boxes(data)
        assert len(result) == 2
        assert result[0]['text'] == 'Line1'
        assert result[1]['text'] == 'Line2'

    def test_custom_tolerances(self):
        """Test with custom x and y tolerances."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Hello'},
            {'position': [60, 20, 100, 30], 'text': 'World'}  # 10 pixels apart
        ]
        
        # With xtol=15, should merge
        result = group_and_merge_boxes(data, xtol=15, ytol=5)
        assert len(result) == 1
        assert result[0]['text'] == 'Hello World'
        
        # With xtol=5, should not merge
        result = group_and_merge_boxes(data, xtol=5, ytol=5)
        assert len(result) == 2

    def test_sorting_within_lines(self):
        """Test that boxes are sorted by x-coordinate within lines."""
        data = [
            {'position': [60, 20, 100, 30], 'text': 'Second'},  # Higher x-coordinate
            {'position': [10, 20, 50, 30], 'text': 'First'}     # Lower x-coordinate
        ]
        result = group_and_merge_boxes(data)
        
        # Should be merged in correct order
        assert len(result) == 1
        assert result[0]['text'] == 'First Second'

    def test_complex_merging_scenario(self):
        """Test complex scenario with multiple lines and merges."""
        data = [
            # Line 1: Three boxes that should merge
            {'position': [10, 20, 30, 30], 'text': 'A'},
            {'position': [35, 20, 55, 30], 'text': 'B'},
            {'position': [60, 20, 80, 30], 'text': 'C'},
            
            # Line 2: Two separate groups
            {'position': [10, 50, 30, 60], 'text': 'D'},
            {'position': [100, 50, 120, 60], 'text': 'E'},  # Far apart, won't merge
            
            # Line 3: Single box
            {'position': [10, 80, 50, 90], 'text': 'F'}
        ]
        
        result = group_and_merge_boxes(data)
        
        # Should have 4 boxes: ABC merged, D alone, E alone, F alone
        assert len(result) == 4
        
        texts = [box['text'] for box in result]
        assert 'A B C' in texts
        assert 'D' in texts
        assert 'E' in texts
        assert 'F' in texts

    def test_bounding_box_calculation(self):
        """Test that merged bounding boxes are calculated correctly."""
        data = [
            {'position': [10, 20, 50, 35], 'text': 'Box1'},
            {'position': [55, 15, 95, 30], 'text': 'Box2'}
        ]
        
        result = group_and_merge_boxes(data)
        assert len(result) == 1
        
        # Merged box should have:
        # x1 = min(10, 55) = 10
        # y1 = min(20, 15) = 15
        # x2 = max(50, 95) = 95
        # y2 = max(35, 30) = 35
        expected_position = [10, 15, 95, 35]
        assert result[0]['position'] == expected_position

    def test_multiple_merges_in_line(self):
        """Test multiple consecutive merges within a single line."""
        data = [
            {'position': [10, 20, 30, 30], 'text': 'A'},
            {'position': [35, 20, 55, 30], 'text': 'B'},
            {'position': [60, 20, 80, 30], 'text': 'C'},
            {'position': [85, 20, 105, 30], 'text': 'D'}
        ]
        
        result = group_and_merge_boxes(data)
        assert len(result) == 1
        assert result[0]['text'] == 'A B C D'

    def test_edge_case_exact_tolerance(self):
        """Test edge case where distance exactly equals tolerance."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Hello'},
            {'position': [75, 20, 115, 30], 'text': 'World'}  # Exactly 25 pixels apart
        ]
        
        # Should merge when distance equals tolerance
        result = group_and_merge_boxes(data, xtol=25)
        assert len(result) == 1
        assert result[0]['text'] == 'Hello World'

    def test_overlapping_boxes(self):
        """Test handling of overlapping boxes."""
        data = [
            {'position': [10, 20, 60, 30], 'text': 'Overlap1'},
            {'position': [40, 20, 90, 30], 'text': 'Overlap2'}  # Overlapping
        ]
        
        result = group_and_merge_boxes(data)
        assert len(result) == 1
        assert result[0]['text'] == 'Overlap1 Overlap2'
        # Merged box should encompass both
        assert result[0]['position'] == [10, 20, 90, 30]

    def test_preserve_non_position_attributes(self):
        """Test that non-position attributes are preserved when possible."""
        data = [
            {'position': [10, 20, 50, 30], 'text': 'Hello', 'font': 'Arial'},
            {'position': [100, 20, 140, 30], 'text': 'World', 'color': 'red'}  # Won't merge
        ]
        
        result = group_and_merge_boxes(data)
        assert len(result) == 2
        
        # Original attributes should be preserved for non-merged boxes
        hello_box = next(box for box in result if 'Hello' in box['text'])
        world_box = next(box for box in result if 'World' in box['text'])
        
        assert hello_box.get('font') == 'Arial'
        assert world_box.get('color') == 'red' 