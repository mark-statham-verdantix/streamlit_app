# streamlit_GQ_app.py - Complete Streamlit Web App
"""
Streamlit web app for GQ Chart Generator
Run with: streamlit run streamlit_GQ_app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import os
import tempfile
import zipfile
from io import BytesIO
import base64
from matplotlib.font_manager import FontProperties

# Page configuration
st.set_page_config(
    page_title="GQ Chart Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
@font-face {
    font-family: 'Gellix';
    src: url('Gellix-Regular.ttf') format('truetype');
    font-weight: normal;
}
html, body, [class*='css'], .stApp, .stTextInput, .stButton, .stSelectbox, .stDataFrame, .stMetric, .stMarkdown, .stHeader, .stSubheader, .stCaption, .stSidebar, .stTabs, .stTab, .stAlert, .stDownloadButton, .stProgress, .stTable, .stRadio, .stCheckbox, .stSlider, .stNumberInput, .stFileUploader, .stForm, .stFormSubmitButton, .stTextArea, .stText, .stTitle, .stContainer, .stColumn, .stExpander, .stTooltip, .stException, .stError, .stWarning, .stSuccess, .stInfo, .stCode, .stJson, .stVegaLiteChart, .stPlotlyChart, .stAltairChart, .stBokehChart, .stPydeckChart, .stGraphvizChart, .stDeckGlChart, .stArrowVegaLiteChart, .stArrowPlotlyChart, .stArrowAltairChart, .stArrowBokehChart, .stArrowPydeckChart, .stArrowGraphvizChart, .stArrowDeckGlChart {
    font-family: 'Gellix', Arial, sans-serif !important;
}
/* Ensure all markdown content uses Gellix */
.stMarkdown, .stMarkdown p, .stMarkdown ul, .stMarkdown ol, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown strong, .stMarkdown em, .stMarkdown code, .stMarkdown pre {
    font-family: 'Gellix', Arial, sans-serif !important;
}
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-family: 'Gellix', Arial, sans-serif !important;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    font-family: 'Gellix', Arial, sans-serif !important;
}
.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border: 1px solid #c3e6cb;
    font-family: 'Gellix', Arial, sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

class StreamlitGQGenerator:
    def __init__(self):
        self.data = None
        self.MAX_CRITERIA_DISPLAY = 8
        
        # Professional color palette - exact match to target
        self.colors = {
            'capabilities': '#4285F4',    # Clean blue
            'momentum': '#54F2BF',        # Bright teal
            'field_line': '#9E9E9E',      # Medium gray for connecting lines
            'field_markers': '#757575',   # Darker gray for min/max markers
            'range_bg': '#F5F5F5',        # Very light gray for range background
            'text_primary': '#212121',    # Dark gray for primary text
            'text_secondary': '#616161',  # Medium gray for secondary text
            'grid': '#E0E0E0',           # Light gray for grid
            'background': '#FFFFFF'       # Pure white background
        }
        
        # Load Gellix font for matplotlib
        try:
            self.custom_font = FontProperties(fname='Gellix-Regular.ttf')
        except Exception:
            self.custom_font = FontProperties(family='Arial', style='normal', weight='normal')
    
    def load_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file type. Please upload CSV or Excel file.")
            
            # Validate required columns
            required_columns = ['vendor', 'sum', 'criteria', 'axis', 'year']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Convert sum to numeric, handling string values
            self.data['sum'] = pd.to_numeric(self.data['sum'], errors='coerce').fillna(0)
            
            return True, "Data loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading file: {str(e)}"
    
    def get_vendors(self):
        """Get list of available vendors"""
        if self.data is not None:
            return sorted(self.data['vendor'].unique())
        return []
    
    def get_years(self):
        """Get list of available years"""
        if self.data is not None:
            return sorted(self.data['year'].unique())
        return []
    
    def get_actual_combinations(self):
        """Get only the vendor/year combinations that actually exist in the data"""
        if self.data is None:
            return []
        
        combinations = []
        grouped = self.data.groupby(['vendor', 'year'])
        
        for (vendor, year), group in grouped:
            # Check if this combination has meaningful data
            caps_data = group[group['axis'].isin(['Capabilities', 'Capability'])]
            momentum_data = group[group['axis'] == 'Momentum']
            
            if len(caps_data) > 0 or len(momentum_data) > 0:
                combinations.append({
                    'vendor': vendor,
                    'year': year,
                    'total_records': len(group),
                    'capabilities_count': len(caps_data),
                    'momentum_count': len(momentum_data)
                })
        
        combinations.sort(key=lambda x: (x['vendor'], x['year']))
        return combinations
    
    def process_axis_data(self, data, axis_types, selected_vendor):
        """Process data for capabilities or momentum"""
        axis_data = data[data['axis'].isin(axis_types)]
        criteria_stats = []
        
        for criteria in axis_data['criteria'].unique():
            criteria_data = axis_data[axis_data['criteria'] == criteria]
            vendor_rows = criteria_data[criteria_data['vendor'] == selected_vendor]
            vendor_score = vendor_rows['sum'].iloc[0] if len(vendor_rows) > 0 else 0
            all_scores = criteria_data['sum'].tolist()
            
            if all_scores:
                criteria_stats.append({
                    'criteria': criteria,
                    'vendor_score': vendor_score,
                    'min_score': min(all_scores),
                    'max_score': max(all_scores),
                    'axis': axis_types[0]
                })
        
        criteria_stats.sort(key=lambda x: x['vendor_score'], reverse=True)
        return criteria_stats[:self.MAX_CRITERIA_DISPLAY]
    
    def create_chart(self, vendor, year):
        """Create chart and return matplotlib figure"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        year_data = self.data[self.data['year'] == year]
        if len(year_data) == 0:
            raise ValueError(f"No data found for year {year}")
        
        if vendor not in year_data['vendor'].unique():
            raise ValueError(f"Vendor '{vendor}' not found in {year} data")
        
        capabilities_data = self.process_axis_data(year_data, ['Capabilities', 'Capability'], vendor)
        momentum_data = self.process_axis_data(year_data, ['Momentum'], vendor)
        
        if not capabilities_data and not momentum_data:
            raise ValueError(f"No capabilities or momentum data found for {vendor} in {year}")
        
        # Create matplotlib figure
        plt.rcParams.update({
            'font.size': 10,
            'axes.linewidth': 0.8,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': False,
            'ytick.right': False,
            'axes.grid': False,
            'grid.alpha': 0.3,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
        fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='white')
        
        # Title and layout
        title_height = 0.12
        legend_height = 0.08
        margin_top = 0.05
        margin_bottom = 0.08
        chart_height = 1 - title_height - legend_height - margin_top - margin_bottom
        
        # Main title
        fig.text(0.5, 0.95, f'{vendor} Performance vs. Field ({year})', 
                fontsize=18, fontweight='normal', ha='center', va='top',
                color=self.colors['text_primary'], fontproperties=self.custom_font)
        
        # Chart areas
        left_margin = 0.08
        right_margin = 0.08
        center_gap = 0.06
        chart_width = (1 - left_margin - right_margin - center_gap) / 2
        chart_bottom = margin_bottom
        momentum_push = 0.1
        
        # Create subplots
        ax1 = fig.add_axes([left_margin, chart_bottom, chart_width, chart_height])
        ax2 = fig.add_axes([left_margin + chart_width + center_gap + momentum_push, chart_bottom, chart_width, chart_height])
        
        # Create charts
        if capabilities_data:
            self._create_GQ(ax1, capabilities_data, 'Top capabilities', self.colors['capabilities'])
        else:
            self._create_empty_chart(ax1, 'Top capabilities')
            
        if momentum_data:
            self._create_GQ(ax2, momentum_data, 'Top momentum', self.colors['momentum'])
        else:
            self._create_empty_chart(ax2, 'Top momentum')
        
        # Create legend
        self._create_legend(fig)
        
        # Add explanatory text below the charts
        fig.text(0.5, 0.0001, 'Charts show top 8 scoring criteria for each vendor', 
                fontsize=9, ha='center', va='bottom',
                color=self.colors['text_secondary'], style='italic', fontproperties=self.custom_font)
        
        plt.tight_layout()
        return fig
    
    def _create_GQ(self, ax, data, chart_type, dot_color):
        """Create GQ chart on axis"""
        if not data:
            return
        
        data = list(reversed(data))
        n_items = len(data)
        y_positions = np.arange(n_items)
        
        ax.set_xlim(0, 3.2)
        ax.set_ylim(-0.5, n_items - 0.5)
        
        # Draw elements
        for i, item in enumerate(data):
            y_pos = y_positions[i]
            
            # Background range
            range_width = item['max_score'] - item['min_score']
            if range_width > 0:
                range_rect = patches.Rectangle(
                    (item['min_score'], y_pos - 0.12), range_width, 0.24,
                    facecolor=self.colors['range_bg'], edgecolor='none', alpha=0.6, zorder=1
                )
                ax.add_patch(range_rect)
            
            # Min/max markers
            marker_height = 0.08
            ax.plot([item['min_score'], item['min_score']], 
                   [y_pos - marker_height, y_pos + marker_height],
                   color=self.colors['field_markers'], linewidth=2, zorder=3)
            ax.plot([item['max_score'], item['max_score']], 
                   [y_pos - marker_height, y_pos + marker_height],
                   color=self.colors['field_markers'], linewidth=2, zorder=3)
            
            # Vendor score
            if item['vendor_score'] != 0:
                ax.scatter([item['vendor_score']], [y_pos], s=135, c=dot_color, 
                          edgecolors=dot_color, linewidths=1.5, alpha=1, zorder=4)
        
        # Configure axes
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['0', '1', '2', '3'], fontsize=9, 
                          color=self.colors['text_secondary'], fontproperties=self.custom_font)
        ax.set_xlabel('Criteria-level score', fontsize=10, 
                     color=self.colors['text_primary'], labelpad=8, fontproperties=self.custom_font)
        
        # Y-axis labels
        y_labels = []
        for item in data:
            criteria = item['criteria'].strip()
            if criteria:
                # Smart case processing - preserve acronyms, sentence case for others
                words = criteria.split(' ')
                processed_words = []
                for i, word in enumerate(words):
                    # Check if word is likely an acronym (all caps or 2-3 letters)
                    if word.isupper() or (len(word) <= 3 and word.isalpha()):
                        processed_words.append(word)  # Keep acronyms as-is
                    else:
                        # Apply sentence case to non-acronyms (first letter only if it's the first word)
                        if i == 0:
                            processed_words.append(word[0].upper() + word[1:].lower())
                        else:
                            processed_words.append(word.lower())
                criteria = ' '.join(processed_words)
            criteria = criteria.replace(' and ', ' & ')
            if len(criteria) > 20:
                words = criteria.split(' ')
                lines = []
                current_line = ''
                for word in words:
                    test_line = current_line + (' ' + word if current_line else word)
                    if len(test_line) <= 20:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                y_labels.append('\n'.join(lines))
            else:
                y_labels.append(criteria)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9, ha='right',
                          color=self.colors['text_primary'], fontproperties=self.custom_font)
        ax.set_title(chart_type, fontsize=12, fontweight='normal',
                    color=self.colors['text_primary'], pad=15, fontproperties=self.custom_font)
        
        # Style spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.colors['text_secondary'])
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_color(self.colors['text_secondary'])
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(left=False, bottom=True, length=3, width=0.5, 
                      color=self.colors['text_secondary'])
    
    def _create_empty_chart(self, ax, chart_type):
        """Create empty chart placeholder"""
        ax.text(0.5, 0.5, f'No {chart_type} data available', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color=self.colors['text_secondary'],
               style='italic', fontproperties=self.custom_font)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)
        ax.set_title(chart_type, fontsize=12, fontweight='normal',
                    color=self.colors['text_primary'], pad=15, fontproperties=self.custom_font)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_legend(self, fig):
        """Create legend"""
        legend_elements = [
            patches.Patch(facecolor=self.colors['range_bg'], edgecolor='none', label='Field range'),
            Line2D([0], [0], color=self.colors['field_line'], lw=2, label='Field min/max'),
            Line2D([0], [0], marker='o', color='w', label='Capabilities',
                   markerfacecolor=self.colors['capabilities'], markersize=10, lw=0),
            Line2D([0], [0], marker='o', color='w', label='Momentum',
                   markerfacecolor=self.colors['momentum'], markersize=10, lw=0),
        ]
        
        leg = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.87),
                  ncol=4, frameon=False, fontsize=9)
        for text in leg.get_texts():
            text.set_fontproperties(self.custom_font)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = StreamlitGQGenerator()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">GQ Chart Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Create publication-quality charts for Adobe InDesign</p>', unsafe_allow_html=True)
    
    # Sidebar for file upload and data info
    with st.sidebar:
        st.header("Data Input")
        
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload Excel or CSV file with columns: vendor, sum, criteria, axis, year"
        )
        
        if uploaded_file is not None:
            if st.button("Load Data", type="primary"):
                with st.spinner("Loading data..."):
                    success, message = st.session_state.generator.load_data(uploaded_file)
                    if success:
                        st.success(message)
                        st.session_state.data_loaded = True
                        
                        # Show data summary
                        data = st.session_state.generator.data
                        actual_combinations = st.session_state.generator.get_actual_combinations()
                        
                        st.markdown("### Data Summary")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Records", len(data))
                            st.metric("Vendors", len(st.session_state.generator.get_vendors()))
                        with col2:
                            st.metric("Years", len(st.session_state.generator.get_years()))
                            st.metric("Valid Combinations", len(actual_combinations))
                            
                    else:
                        st.error(message)
                        st.session_state.data_loaded = False
        
        # Show sample data format
        if not st.session_state.data_loaded:
            st.markdown("### Required Data Format")
            sample_data = pd.DataFrame({
                'vendor': ['Vendor A', 'Vendor A', 'Vendor A', 'Vendor B', 'Vendor B', 'Vendor B','Vendor C', 'Vendor C', 'Vendor C'],
                'sum': [2.5, 1.8, 2.1,2.0, 1.2, 2.3,2.4, 1.0, 2.3],
                'criteria': ['Market Share', 'Market Share', 'Market Share', 'Market Share', 'Market Share', 'Market Share', 'Market Share', 'Market Share', 'Market Share'],
                'axis': ['Momentum', 'Momentum', 'Momentum', 'Momentum', 'Momentum', 'Momentum', 'Momentum', 'Momentum', 'Momentum'],
                'year': [2025, 2025, 2025,2025,2025,2025,2025,2025,2025]
            })
            st.dataframe(sample_data, use_container_width=True)
            st.caption("Your file should have these exact column names")
    
    # Main content area
    if st.session_state.data_loaded:
        data = st.session_state.generator.data
        vendors = st.session_state.generator.get_vendors()
        years = st.session_state.generator.get_years()
        actual_combinations = st.session_state.generator.get_actual_combinations()
        
        # Chart generation options
        st.markdown("## Chart Options")
        
        tab1, tab2, tab3 = st.tabs(["Single Chart", "Batch Creation", "Data Preview"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_vendor = st.selectbox("Select Vendor", vendors, key="single_vendor")
            with col2:
                selected_year = st.selectbox("Select Year", years, key="single_year")
            if st.button("Generate Single Chart", type="primary"):
                try:
                    with st.spinner(f"Creating chart for {selected_vendor} ({selected_year})..."):
                        fig = st.session_state.generator.create_chart(selected_vendor, selected_year)
                        
                        # Display chart
                        st.pyplot(fig)
                        
                        # Download options
                        st.markdown("### Download Chart")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # PNG download
                            img_buffer = BytesIO()
                            fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                                       facecolor='white', dpi=300)
                            img_buffer.seek(0)
                            
                            st.download_button(
                                label="Download PNG",
                                data=img_buffer.getvalue(),
                                file_name=f"{selected_vendor}_{selected_year}_chart.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            # SVG download
                            svg_buffer = BytesIO()
                            fig.savefig(svg_buffer, format='svg', bbox_inches='tight', facecolor='white')
                            svg_buffer.seek(0)
                            
                            st.download_button(
                                label="Download SVG",
                                data=svg_buffer.getvalue(),
                                file_name=f"{selected_vendor}_{selected_year}_chart.svg",
                                mime="image/svg+xml"
                            )
                        
                        with col3:
                            # PDF download
                            pdf_buffer = BytesIO()
                            fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight', facecolor='white')
                            pdf_buffer.seek(0)
                            
                            st.download_button(
                                label="Download PDF",
                                data=pdf_buffer.getvalue(),
                                file_name=f"{selected_vendor}_{selected_year}_chart.pdf",
                                mime="application/pdf"
                            )
                        
                        plt.close(fig)
                        
                        st.markdown("""
                        <div class="success-message">
                        <strong>Chart created successfully!</strong><br>
                        PNG: High-resolution preview<br>
                        SVG: Vector format for Adobe InDesign<br>
                        PDF: Print-ready format
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
        
        with tab2:
            st.markdown("### Batch Chart Creation")
            
            # Show batch preview
            all_possible = len(vendors) * len(years)
            st.info(f"Smart Batch Processing: Will create {len(actual_combinations)} charts from actual data combinations (not {all_possible} possible combinations)")
            
            # Preview table
            if actual_combinations:
                preview_df = pd.DataFrame(actual_combinations)
                st.markdown("Combinations to be processed:")
                st.dataframe(
                    preview_df[['vendor', 'year', 'total_records', 'capabilities_count', 'momentum_count']], 
                    use_container_width=True
                )
            
            if st.button("Create All Charts", type="primary"):
                if not actual_combinations:
                    st.warning("No valid vendor/year combinations found in your data.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Create zip file for all charts
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for i, combo in enumerate(actual_combinations):
                                vendor = combo['vendor']
                                year = combo['year']
                                
                                progress = (i + 1) / len(actual_combinations)
                                progress_bar.progress(progress)
                                status_text.text(f"Creating chart {i+1}/{len(actual_combinations)}: {vendor} ({year})")
                                
                                try:
                                    fig = st.session_state.generator.create_chart(vendor, year)
                                    
                                    # Clean vendor name for filename
                                    clean_vendor = "".join(c for c in vendor if c.isalnum() or c in (' ', '-', '_')).rstrip()
                                    clean_vendor = clean_vendor.replace(' ', '_')
                                    
                                    # Save PNG to zip
                                    png_buffer = BytesIO()
                                    fig.savefig(png_buffer, format='png', bbox_inches='tight', 
                                               facecolor='white', dpi=300)
                                    zip_file.writestr(f"png/{clean_vendor}_{year}_chart.png", png_buffer.getvalue())
                                    
                                    # Save SVG to zip
                                    svg_buffer = BytesIO()
                                    fig.savefig(svg_buffer, format='svg', bbox_inches='tight', facecolor='white')
                                    zip_file.writestr(f"svg/{clean_vendor}_{year}_chart.svg", svg_buffer.getvalue())
                                    
                                    # Save PDF to zip
                                    pdf_buffer = BytesIO()
                                    fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight', facecolor='white')
                                    zip_file.writestr(f"pdf/{clean_vendor}_{year}_chart.pdf", pdf_buffer.getvalue())
                                    
                                    plt.close(fig)
                                    
                                except Exception as e:
                                    st.warning(f"Failed to create chart for {vendor} ({year}): {str(e)}")
                        
                        status_text.text("All charts created successfully!")
                        progress_bar.progress(1.0)
                        
                        # Download zip file
                        st.download_button(
                            label="Download All Charts (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="all_GQ_charts.zip",
                            mime="application/zip"
                        )
                        
                        st.success(f"Successfully created {len(actual_combinations)} charts! Each chart includes PNG, SVG, and PDF formats.")
                        
                    except Exception as e:
                        st.error(f"Error generating charts: {str(e)}")
        
        with tab3:
            st.markdown("### Data Preview")
            
            # Show full data
            st.dataframe(data, use_container_width=True)
            
            # Data analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Vendors by Year:")
                vendor_year_counts = data.groupby(['vendor', 'year']).size().reset_index(name='count')
                st.dataframe(vendor_year_counts, use_container_width=True)
            
            with col2:
                st.markdown("Axis Distribution:")
                axis_counts = data['axis'].value_counts()
                st.bar_chart(axis_counts)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to GQ Chart Generator
        
        This tool creates publication-quality GQ charts for Adobe InDesign. Perfect for market research and competitive analysis visualizations.
        
        ### Getting Started:
        1. Upload your data file (Excel or CSV) using the sidebar
        2. Load the data to validate and preview
        3. Generate charts individually or in batch
        4. Download in multiple formats (PNG, SVG, PDF)
        
        ### Features:
        - Publication-quality output optimized for Adobe InDesign
        - Smart batch processing - only creates charts for actual data combinations
        - Multiple formats - PNG (preview), SVG (vector), PDF (print)
        - Professional styling - exact color matching and typography
        
        """)

if __name__ == "__main__":
    main()