import pandas as pd
import plotly.express as px
from IPython.display import HTML, display
from collections import Counter
import numpy as np

class Visualizer:
    def __init__(self, data, fields_schema: dict):
        # Make a copy to avoid SettingWithCopyWarning if df was a view
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        elif isinstance(data, list):
            self.df = pd.DataFrame(data)
        else:
            raise TypeError("Input data must be a pandas DataFrame or a list of records.")
        self.fields_schema = fields_schema
        
        # If created_utc is present and numeric, convert to datetime once.
        if 'created_utc' in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df['created_utc']):
                # Attempt conversion
                try:
                    self.df.loc[:, 'created_utc'] = pd.to_datetime(self.df['created_utc'], unit='s', errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert 'created_utc' to datetime: {e}")
            # If conversion fails or it's non-numeric and non-datetime, leave as is.
            # The plotting method will handle missing or invalid values gracefully.
        
    def display_section(self, title: str):
        display(HTML(f'<div style="margin:10px 0;"><h4 style="color:#333;margin:0;padding:5px 0;border-bottom:1px solid #ccc;">{title}</h4></div>'))

    def plot_all_fields(self, show_examples=True, save=False, metadata_fields=None, record_fields=None, title_field=None, extra_fields=None):
        if extra_fields:
            record_fields = (record_fields or []) + extra_fields

        for field_name, field_info in self.fields_schema.items():
            ftype = field_info['type']
            rf = (record_fields or []).copy()
            if field_name not in rf:
                rf.append(field_name)

            if ftype == 'boolean':
                self._plot_binary_distribution(field_name, show_examples, save, metadata_fields, rf)
            elif ftype == 'integer':
                self._plot_integer_distribution(field_name, show_examples, save, metadata_fields, rf)
            elif ftype == 'array':
                self._plot_list_field(field_name, show_examples, save, metadata_fields, rf)

    def _plot_binary_distribution(self, field_name, show_examples, save, metadata_fields, record_fields):
        if field_name not in self.df.columns:
            print(f"Field '{field_name}' not found in DataFrame.")
            return
        counts = self.df[field_name].value_counts(dropna=False)
        if counts.empty:
            print(f"No data for '{field_name}' to plot binary distribution.")
            return

        title = f"Distribution of {field_name}"
        percentages = (counts / len(self.df)) * 100
        fig = px.pie(values=percentages.values, names=percentages.index.astype(str), title=title)
        fig.update_traces(texttemplate='%{value:.1f}%')
        self._save_fig(fig, title, save)
        fig.show()

        if show_examples:
            for value in counts.index:
                subset = self.df[self.df[field_name] == value]
                if len(subset) > 0:
                    self.display_section(f"Example Samples for {field_name} = {value}")
                    example_posts = subset.sample(min(3, len(subset)))
                    self._display_samples(example_posts, metadata_fields, record_fields)

    def _plot_integer_distribution(self, field_name, show_examples, save, metadata_fields, record_fields):
        if field_name not in self.df.columns:
            print(f"Field '{field_name}' not found in DataFrame.")
            return
        valid_data = self.df[self.df[field_name].notnull()]
        if valid_data.empty:
            print(f"No valid data for '{field_name}' to plot integer distribution.")
            return

        title = f"{field_name} Distribution"
        fig = px.histogram(valid_data, x=field_name, title=title, nbins=10)
        fig.update_traces(histnorm='percent')
        
        mean_val = valid_data[field_name].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_val:.2f}",
                      annotation_position="top right")
        
        fig.update_layout(yaxis_title="Percent")
        self._save_fig(fig, title, save)
        fig.show()

        if show_examples:
            if len(valid_data) > 0:
                self.display_section(f"Example Samples with {field_name}")
                example_posts = valid_data.sample(min(3, len(valid_data)))
                self._display_samples(example_posts, metadata_fields, record_fields)

    def _plot_list_field(self, field_name, show_examples, save, metadata_fields, record_fields):
        if field_name not in self.df.columns:
            print(f"Field '{field_name}' not found in DataFrame.")
            return
        all_items = []
        for val in self.df[field_name].dropna():
            if isinstance(val, list):
                all_items.extend(val)
        
        if not all_items:
            print(f"No list items found for '{field_name}'.")
            return

        title = f"Most Common {field_name.capitalize()}"
        item_counts = Counter(all_items).most_common(10)
        df_counts = pd.DataFrame(item_counts, columns=[field_name, 'count'])
        if df_counts.empty:
            print(f"No data to plot for '{field_name}'.")
            return
        total_posts = len(self.df)
        df_counts['percent'] = df_counts['count'].apply(lambda x: (x / total_posts) * 100)
        
        fig = px.bar(df_counts, x=field_name, y='percent', title=title)
        fig.update_layout(yaxis_title="Percent of Posts")
        self._save_fig(fig, title, save)
        fig.show()

        if show_examples and not df_counts.empty:
            top_item = df_counts[field_name].iloc[0]
            subset = self.df[self.df[field_name].apply(lambda x: isinstance(x, list) and top_item in x)]
            if len(subset) > 0:
                self.display_section(f"Example Samples with {field_name}")
                example_posts = subset.sample(min(3, len(subset)))
                self._display_samples(example_posts, metadata_fields, record_fields)

    def plot_by_time(self, time_field: str, title: str, freq='M', save=False):
        if time_field not in self.df.columns:
            print(f"{time_field} not in DataFrame columns.")
            return

        valid_times = self.df[self.df[time_field].notnull()].copy()
        if valid_times.empty:
            print(f"No valid values in {time_field} to plot over time.")
            return

        # Try to convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(valid_times[time_field]):
            try:
                valid_times[time_field] = pd.to_datetime(valid_times[time_field])
            except Exception as e:
                print(f"Could not convert '{time_field}' to datetime for plotting: {e}")
                return

        try:
            counts = valid_times[time_field].dt.to_period(freq).value_counts().sort_index()
        except Exception as e:
            print(f"Error processing datetime data: {e}")
            return

        if counts.empty:
            print(f"No data after grouping by period {freq} for {time_field}.")
            return

        fig = px.line(x=counts.index.astype(str), y=counts.values, title=title, labels={'x': 'Time', 'y': 'Count'})
        self._save_fig(fig, title, save)
        fig.show()

    def plot_correlation(self, numeric_fields: list, title="Correlation Matrix", save=False):
        if not numeric_fields:
            print("No numeric fields provided for correlation.")
            return
        numeric_df = self.df[numeric_fields].dropna()
        if numeric_df.empty:
            print("No valid numeric data for correlation.")
            return
        corr = numeric_df.corr()
        if corr.empty or corr.isna().all().all():
            print("Correlation matrix is empty or invalid.")
            return
        fig = px.imshow(corr, text_auto=True, title=title, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        self._save_fig(fig, title, save)
        fig.show()

    def plot_group_comparison(self, group_field: str, value_field: str, agg='mean', title=None, save=False):
        if group_field not in self.df.columns or value_field not in self.df.columns:
            print(f"Group field '{group_field}' or value field '{value_field}' not found.")
            return
        if agg not in ['mean', 'count', 'sum', 'median']:
            agg = 'mean'
        grouped = self.df.groupby(group_field)[value_field]
        if agg == 'mean':
            result = grouped.mean().dropna()
        elif agg == 'count':
            result = grouped.count()
        elif agg == 'sum':
            result = grouped.sum().dropna()
        elif agg == 'median':
            result = grouped.median().dropna()

        if result.empty:
            print(f"No data after aggregation {agg} for {value_field} by {group_field}.")
            return
        t = title or f"{agg.capitalize()} of {value_field} by {group_field}"
        fig = px.bar(x=result.index.astype(str), y=result.values, title=t, labels={'x':group_field,'y':f"{agg}({value_field})"})
        self._save_fig(fig, t, save)
        fig.show()

    def show_samples(self, n=5, metadata_fields=None, record_fields=None, extra_fields=None):
        if extra_fields:
            record_fields = (record_fields or []) + extra_fields

        if self.df.empty:
            print("No samples to display.")
            return
        samples = self.df.sample(min(n, len(self.df)))
        self.display_section(f"Showing {len(samples)} Random Samples")
        self._display_samples(samples, metadata_fields, record_fields)

    def _display_samples(self, samples, metadata_fields, record_fields):
        html_output = '<div style="display:flex;flex-wrap:wrap;">'
        for _, post in samples.iterrows():
            html_output += self.format_sample(post, metadata_fields=metadata_fields, record_fields=record_fields)
        html_output += "</div>"
        display(HTML(html_output))

    def format_sample(self, post, metadata_fields=None, record_fields=None):
        # Show title, text, context at top
        # Show ID, metadata fields, and record fields at bottom
        title = str(post.get('title', 'No title'))
        url = str(post.get('url', ''))
        text = str(post.get('text', ''))
        context_text = str(post.get('context_text', ''))

        html_parts = []
        html_parts.append("<div style='border:1px solid #ddd; border-radius:8px; padding:15px; margin:10px 5px; background:#f9f9f9; display:inline-block; vertical-align:top; width:320px; margin-bottom:10px;'>")

        # Title/URL
        if url and url != 'nan':
            html_parts.append(f"<div style='color:#333; font-size:1.1em; font-weight:bold; margin-bottom:10px;'><a href='{url}' target='_blank' style='text-decoration:none; color:inherit;'>{title}</a></div>")
        else:
            html_parts.append(f"<div style='color:#333; font-size:1.1em; font-weight:bold; margin-bottom:10px;'>{title}</div>")

        # Text/Context
        if text and text != 'nan':
            html_parts.append(f"<div style='margin-bottom:10px;'><strong>Text:</strong><br>{self._truncate_text(text)}</div>")
        if context_text and context_text != 'nan':
            html_parts.append(f"<div style='margin-bottom:10px;'><strong>Context:</strong><br>{self._truncate_text(context_text)}</div>")

        # Metadata/Record fields/ID at bottom
        meta_info = []
        if metadata_fields:
            for mf in metadata_fields:
                if mf in post:
                    val = post[mf]
                    if self._is_valid_value(val):
                        meta_info.append(f"{mf}: {self._convert_value_to_str(val)}")

        rec_info = []
        if record_fields:
            for rf in record_fields:
                if rf in post:
                    val = post[rf]
                    if self._is_valid_value(val):
                        rec_info.append(f"{rf}: {self._convert_value_to_str(val)}")

        id_str = f"ID: {post['id']}"
        bottom_info = []
        if rec_info:
            bottom_info.append(' | '.join(rec_info))
        if meta_info:
            bottom_info.append(' | '.join(meta_info))
        bottom_info.append(id_str)

        if bottom_info:
            html_parts.append(f"<div style='color:#666; margin-top:10px; font-size:0.9em;'>{' | '.join(bottom_info)}</div>")

        html_parts.append("</div>")
        return "".join(html_parts)

    def _is_valid_value(self, val):
        return not (val is None or (isinstance(val, float) and np.isnan(val)))

    def _convert_value_to_str(self, val):
        if isinstance(val, (list, np.ndarray)):
            return ", ".join(map(str, val))
        return str(val)

    def _truncate_text(self, text, length=500):
        return text[:length] + '...' if len(text) > length else text

    def _save_fig(self, fig, title, save):
        if save:
            import os
            if not os.path.exists('plots'):
                os.makedirs('plots')
            fig.write_image(f"plots/{title.lower().replace(' ', '_')}.png")
