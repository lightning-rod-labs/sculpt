from typing import List, Dict, Any, Callable, Optional
import logging
from sculptor.sculptor import Sculptor
from sculptor.utils import load_config

class SculptorPipeline:
    """
    Chains multiple Sculptors together with optional filtering between steps.

    Example usage:
        # Basic usage
        pipeline = SculptorPipeline()
        pipeline.add(sculptor1, lambda x: x['confidence'] > 0.8)
        pipeline.add(sculptor2)
        results = pipeline.process(data)

        # Method chaining
        results = (SculptorPipeline()
            .add(sculptor1, lambda x: x['confidence'] > 0.8)
            .add(sculptor2)
            .add(sculptor3, lambda x: x['type'] == 'relevant')
            .process(data))

        # From config
        pipeline = SculptorPipeline.from_config('pipeline_config.yml')
        results = pipeline.process(data)
    """
    def __init__(self):
        self.steps = []

    def add(self, sculptor: Sculptor, filter_fn: Optional[Callable] = None) -> 'SculptorPipeline':
        """
        Adds a sculptor and optional filter to the pipeline.
        
        Args:
            sculptor: Sculptor instance to process data
            filter_fn: Optional function that takes a dict and returns bool to filter records
        """
        self.steps.append((sculptor, filter_fn))
        return self  # Enable method chaining

    @classmethod
    def from_config(cls, config_path: str) -> 'SculptorPipeline':
        """Creates a pipeline from a config file."""
        config = load_config(config_path)
        
        pipeline = cls()
        for step in config.get('steps', []):
            sculptor = Sculptor(**step['sculptor'])
            filter_fn = eval(step.get('filter')) if 'filter' in step else None
            pipeline.add(sculptor, filter_fn)
            
        return pipeline

    def get_schema_fields(self) -> Dict[str, Dict[str, Any]]:
        """Gets a combined dictionary of all schema fields from all sculptors in the pipeline."""
        schema_fields = {}
        for sculptor, _ in self.steps:
            schema_fields.update(sculptor.schema)
        return schema_fields

    def process(
        self,
        data: List[Dict[str, Any]],
        n_workers: int = 1,
        show_progress: bool = True,
        merge_input: bool = True,
    ) -> List[Dict[str, Any]]:
        """Processes data through all sculptors in the pipeline."""
        # If input is a DataFrame, convert it to list of dicts
        if hasattr(data, 'to_dict'):
            data = data.to_dict('records')
        
        current_data = data
        
        for i, (sculptor, filter_fn) in enumerate(self.steps, 1):
            if show_progress:
                logging.info(f"Step {i}/{len(self.steps)}")
            
            # Process current batch through sculptor
            results = sculptor.sculpt_batch(
                current_data,
                n_workers=n_workers,
                show_progress=show_progress,
                merge_input=merge_input  # This maintains original fields + new ones
            )
            # Apply filter if provided
            if filter_fn:
                # Keep results where filter condition is True
                current_data = [
                    result for result in results
                    if filter_fn(result)
                ]
                if show_progress:
                    logging.info(f"Filtered to {len(current_data)} items")
            else:
                current_data = results
                
        return current_data