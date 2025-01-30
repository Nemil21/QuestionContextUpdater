import streamlit as st
import pandas as pd
import google.generativeai as genai
from typing import Dict, List
import os
from io import BytesIO
import time
from datetime import datetime

class QuestionProcessor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate_prompt(self, row: Dict) -> str:
        return f"""Please modify the following question and its options to include specific context about {row['State']} and the attribute: {row['Attribute']}

Original Question: {row['Question']}


Original Options:
1. {row['Option1']}
2. {row['Option2']}
3. {row['Option3']}
4. {row['Option4']}

Please provide:
1. A corrected version of the question with {row['State']} and {row['Attribute']} context. Make sure to include some kind of a cultural or regional artifact in the question.
2. Modified options that reflect this context but must have the right answer as {row['State']} as i dont want confusing options. if you use {row['State']} as the right answer for the question form it in a way like which region...? and use other states of india as options and when using other three form it like which culture...? use other cultures as options
3. The correct answer with explanation
4. Relevant citations or sources but DO NOT HALLUCINATE or make up information only use word to word citation and try to use wikipedia for citations.
5. Make sure to include an {row['State']} specific cultural artifact that is UNIQUE to {row['State']}. DO NOT HALLUCINATE about the artifact.

Format your response as follows:
CORRECTED_QUESTION: [Your modified question]
CORRECTED_OPTIONS:
1. [Modified option 1]
2. [Modified option 2]
3. [Modified option 3]
4. [Modified option 4]
CORRECT_ANSWER: [correct option]
EXPLANATION: [Detailed explanation]
CITATIONS: [Relevant sources]"""

    def process_question(self, row: Dict) -> List[Dict]:
        try:
            prompt = self.generate_prompt(row)
            responses = [self.model.generate_content(prompt) for _ in range(3)]
            suggestions = [self.parse_llm_response(response.text) for response in responses]
            return suggestions
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return []

    def parse_llm_response(self, response: str) -> Dict:
        try:
            lines = response.split('\n')
            sections = {
                'Corrected_Question': '',
                'Corrected_Options': [],
                'Corrected_Answer': '',
                'Full Answer': '',
                'Answer Source': ''
            }
            
            current_section = None
            
            for line in lines:
                if line.startswith('CORRECTED_QUESTION:'):
                    current_section = 'Corrected_Question'
                    sections['Corrected_Question'] = line.replace('CORRECTED_QUESTION:', '').strip()
                elif line.startswith('CORRECTED_OPTIONS:'):
                    current_section = 'Corrected_Options'
                elif line.startswith('CORRECT_ANSWER:'):
                    current_section = 'Corrected_Answer'
                    sections['Corrected_Answer'] = line.replace('CORRECT_ANSWER:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    current_section = 'Full Answer'
                    sections['Full Answer'] = line.replace('EXPLANATION:', '').strip()
                elif line.startswith('CITATIONS:'):
                    current_section = 'Answer Source'
                    sections['Answer Source'] = line.replace('CITATIONS:', '').strip()
                elif line.strip() and current_section == 'Corrected_Options':
                    if line.strip().startswith(('1.', '2.', '3.', '4.')):
                        sections['Corrected_Options'].append(line.strip())
                elif line.strip() and current_section in ['Full Answer', 'Answer Source']:
                    sections[current_section] += ' ' + line.strip()
            
            return sections
        except Exception as e:
            st.error(f"Error parsing LLM response: {str(e)}")
            return {
                'Corrected_Question': '',
                'Corrected_Options': [],
                'Corrected_Answer': '',
                'Full Answer': '',
                'Answer Source': ''
            }

def load_data(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.error(f"Error reading CSV with {encoding} encoding: {str(e)}")
                    return None
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        required_columns = ['State', 'Attribute', 'Question', 
                          'Option1', 'Option2', 'Option3', 'Option4']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        return df.to_dict('records')
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None
def main():
    st.set_page_config(page_title="Question Context Updater", layout="wide")
    st.title("Question Context Updater")

    # Initialize session state with better persistence
    if 'data' not in st.session_state:
        st.session_state.data = None
        st.session_state.original_file = None
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'editing' not in st.session_state:
        st.session_state.editing = False
    # Modify the initialization section of your code
    if 'edit_data' not in st.session_state:
        st.session_state.edit_data = {
            'question': '',
            'options': ['', '', '', ''],
            'answer': '',
            'explanation': '',
            'citations': ''
        }
    if 'changes_made' not in st.session_state:
        st.session_state.changes_made = False

    # API Key input
    api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
    if api_key:
        st.session_state.processor = QuestionProcessor(api_key)

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload File", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        if 'original_file' not in st.session_state or st.session_state.original_file != uploaded_file.name:
            data = load_data(uploaded_file)
            if data:
                st.session_state.data = data
                st.session_state.original_file = uploaded_file.name
                st.session_state.changes_made = False
                st.sidebar.success(f"File uploaded successfully! {len(data)} questions loaded.")

    if st.session_state.data and st.session_state.processor:
        if st.session_state.data and st.session_state.processor:
        # Advanced Navigation Controls
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.subheader("Navigation")
                    # Jump to question dropdown
                    total_questions = len(st.session_state.data)
                    jump_to = st.number_input(
                        "Jump to question number",
                        min_value=1,
                        max_value=total_questions,
                        value=st.session_state.current_index + 1,
                        key="jump_input"
                    )
                    if jump_to != st.session_state.current_index + 1:
                        st.session_state.current_index = jump_to - 1
                        st.session_state.editing = False
                        st.rerun()
                    
                    # Previous/Next buttons
                    prev_col, next_col = st.columns(2)
                    with prev_col:
                        if st.button("◀ Previous", key="prev_btn", disabled=st.session_state.current_index == 0):
                            st.session_state.current_index -= 1
                            st.session_state.editing = False
                    with next_col:
                        if st.button("Next ▶", key="next_btn", disabled=st.session_state.current_index == total_questions - 1):
                            st.session_state.current_index += 1
                            st.session_state.editing = False
                
                with col2:
                    st.subheader("Questions Overview")
                    # Search functionality
                    search_term = st.text_input("Search questions", key="search_input")
                    
                    # Create DataFrame for preview
                    preview_df = pd.DataFrame(st.session_state.data)
                    if search_term:
                        mask = preview_df['Question'].str.contains(search_term, case=False, na=False)
                        filtered_df = preview_df[mask]
                    else:
                        filtered_df = preview_df
                    
                    # Display clickable question preview
                    st.dataframe(
                        filtered_df[['Question', 'State', 'Attribute']].reset_index(),
                        use_container_width=True,
                        height=150
                    )
                    
                    if st.session_state.current_index < total_questions:
                        st.info(f"Currently viewing Question {st.session_state.current_index + 1} of {total_questions}")

            # Show current question status
            st.markdown("---")

            current_row = st.session_state.data[st.session_state.current_index]

        # Display current question
        with st.expander("Current Question Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("*State:*", current_row['State'])
            with col2:
                st.write("*Attribute:*", current_row['Attribute'])
            
            st.write("*Original Question:*", current_row['Question'])

            st.write("*Original Options:*")
            for i in range(1, 5):
                st.write(f"{i}. {current_row[f'Option{i}']}")

            st.write("*Original Full Answer:*", current_row['Full Answer'])
            
        # Process button
        if st.button("Process Current Question", key="process_btn"):
            with st.spinner("Processing..."):
                st.session_state.suggestions = st.session_state.processor.process_question(current_row)
        
        # Display suggestions with editing interface
        if st.session_state.suggestions:
            with st.expander("Generated Suggestions", expanded=True):
                option = st.selectbox("Choose a suggested question:", 
                                    options=[f"Suggestion {i+1}" for i in range(len(st.session_state.suggestions))],
                                    key="suggestion_select")
                selected_index = int(option.split(' ')[-1]) - 1
                selected_suggestion = st.session_state.suggestions[selected_index]
                
                # Display the selected suggestion
                if not st.session_state.editing:
                    st.write("*Suggested Question:*", selected_suggestion['Corrected_Question'])
                    st.write("*Suggested Options:*")
                    for opt in selected_suggestion['Corrected_Options']:
                        st.write(opt)
                    st.write("*Explanation:*", selected_suggestion['Full Answer'])
                    st.write("*Citations:*", selected_suggestion['Answer Source'])
                
                    # Add edit and direct apply buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Edit This Suggestion", key="edit_btn"):
                            st.session_state.editing = True
                            
                            st.session_state.edit_data = {
                                'question': selected_suggestion['Corrected_Question'],
                                'options': [opt.split('. ')[1] if '. ' in opt else opt 
                                        for opt in selected_suggestion['Corrected_Options']],
                                'answer': selected_suggestion['Corrected_Answer'],
                                'explanation': selected_suggestion['Full Answer'],
                                'citations': selected_suggestion['Answer Source']
                            }
                            st.rerun()
                    
                    with col2:
                        if st.button("Use Without Editing", key="use_without_edit_btn"):
                            updated_row = current_row.copy()
                            updated_row.update({
                                'Question': selected_suggestion['Corrected_Question'],
                                'Option1': selected_suggestion['Corrected_Options'][0].split('. ')[1] if selected_suggestion['Corrected_Options'] else '',
                                'Option2': selected_suggestion['Corrected_Options'][1].split('. ')[1] if len(selected_suggestion['Corrected_Options']) > 1 else '',
                                'Option3': selected_suggestion['Corrected_Options'][2].split('. ')[1] if len(selected_suggestion['Corrected_Options']) > 2 else '',
                                'Option4': selected_suggestion['Corrected_Options'][3].split('. ')[1] if len(selected_suggestion['Corrected_Options']) > 3 else '',
                                'Answer': selected_suggestion['Corrected_Answer'],
                                'Full Answer': selected_suggestion['Full Answer'],
                                'Answer Source': selected_suggestion['Answer Source']
                            })
                            st.session_state.data[st.session_state.current_index] = updated_row
                            st.session_state.changes_made = True
                            st.success("Suggestion applied successfully!")
                            st.rerun()

                # Show editing interface if editing is active
                if st.session_state.editing:
                    st.subheader("Edit Suggestion")
                    edited_question = st.text_area(
                        "Edit Question",
                        value=st.session_state.edit_data['question'],
                        height=100,
                        key="edit_question"
                    )
                    
                    st.subheader("Edit Options")
                    edited_options = []
                    col1, col2 = st.columns(2)
                    with col1:
                        for i in range(2):
                            edited_opt = st.text_area(
                                f"Edit Option {i+1}",
                                value=st.session_state.edit_data['options'][i],
                                height=70,
                                key=f"edit_opt_{i}"
                            )
                            edited_options.append(edited_opt)
                    with col2:
                        for i in range(2, 4):
                            edited_opt = st.text_area(
                                f"Edit Option {i+1}",
                                value=st.session_state.edit_data['options'][i],
                                height=70,
                                key=f"edit_opt_{i}"
                            )
                            edited_options.append(edited_opt)
                    
                    # Move radio button outside the column loop
                    st.subheader("Select Correct Answer")
                    edited_answer = st.radio(
                        "Choose the correct answer",
                        options=edited_options,
                        index=edited_options.index(st.session_state.edit_data['answer']) if st.session_state.edit_data['answer'] in edited_options else 0,
                        key="edit_answer_radio"  # Changed key to be unique
                    )
                    
                    edited_explanation = st.text_area(
                        "Edit Explanation",
                        value=st.session_state.edit_data['explanation'],
                        height=150,
                        key="edit_explanation"
                    )
                    
                    edited_citations = st.text_area(
                        "Edit Citations",
                        value=st.session_state.edit_data['citations'],
                        height=100,
                        key="edit_citations"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save Changes", key="save_changes_btn"):
                            updated_row = current_row.copy()
                            updated_row.update({
                                'Question': edited_question,
                                'Option1': edited_options[0],
                                'Option2': edited_options[1],
                                'Option3': edited_options[2],
                                'Option4': edited_options[3],
                                'Answer': edited_answer,  # This will now store the actual selected option text
                                'Full Answer': edited_explanation,
                                'Answer Source': edited_citations
                            })
                            st.session_state.data[st.session_state.current_index] = updated_row
                            st.session_state.changes_made = True
                            st.session_state.editing = False
                            st.success("Changes saved successfully!")
                            st.rerun()
                    
                    with col2:
                        if st.button("Cancel", key="cancel_btn"):
                            st.session_state.editing = False
                            st.rerun()

        # Export functionality
        if st.session_state.data is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Export Data")
            
            if st.session_state.changes_made:
                export_format = st.sidebar.radio("Choose export format:", ["CSV", "Excel"], key="export_format")
                
                if st.sidebar.button("Export Updated Data", key="export_btn"):
                    try:
                        df = pd.DataFrame(st.session_state.data)
                        timestamp = "20250112_072058"
                        
                        if export_format == "CSV":
                            csv = df.to_csv(index=False).encode('utf-8')
                            filename = f"updated_questions_{timestamp}.csv"
                            
                            st.sidebar.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=filename,
                                mime='text/csv',
                                key='download_csv'
                            )
                            
                        else:  # Excel format
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False)
                            
                            filename = f"updated_questions_{timestamp}.xlsx"
                            excel_data = output.getvalue()
                            
                            st.sidebar.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=filename,
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                key='download_excel'
                            )
                        
                        st.sidebar.success("Export prepared successfully!")
                        with st.sidebar.expander("Preview Updated Data"):
                            st.dataframe(df.head())
                            
                    except Exception as e:
                        st.sidebar.error(f"Error during export: {str(e)}")
            else:
                st.sidebar.info("No changes have been made yet. Make some changes before exporting.")

if __name__ == '__main__':
    main()
