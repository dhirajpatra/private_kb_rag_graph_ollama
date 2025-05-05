# ui/file_table.py
import streamlit as st

def render_file_table():
    st.markdown("#### Case Files")
    files = [
        {"name": "LOI_NovaTech_Stratocore.pdf", "type": "Agreement", "topic": "Deal initiation"},
        {"name": "MIPA_FinalDraft.docx", "type": "Agreement", "topic": "Purchase terms"},
        {"name": "Email_Thread_IP_Rights.msg", "type": "Correspondence", "topic": "IP ownership dispute"},
        {"name": "Notice_BreachOfWarranty.docx", "type": "Summons", "topic": "Breach allegations"},
        {"name": "WitnessStatement.pdf", "type": "Evidence", "topic": "Financial representations"},
        {"name": "Settlement_Proposal.msg", "type": "Correspondence", "topic": "Settlement proposal"},
    ]
    cols = st.columns([3, 2, 3])
    cols[0].markdown("**📄 File**")
    cols[1].markdown("**🏷️ Type**")
    cols[2].markdown("**💬 Topic**")
    for f in files:
        cols = st.columns([3, 2, 3])
        cols[0].markdown(f"📎 {f['name']}")
        cols[1].markdown(f"`{f['type']}`")
        cols[2].markdown(f"`{f['topic']}`")
