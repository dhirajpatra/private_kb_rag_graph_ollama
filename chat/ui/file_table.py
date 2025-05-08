# ui/file_table.py
import streamlit as st

def render_file_table():
    st.markdown("#### Case Files")
    files = [
        {"name": "LOI_TataReliance_Tieup.pdf", "type": "Agreement", "topic": "Business collaboration"},
        {"name": "SPA_FinalDraft_HDFC_ICICI.docx", "type": "Agreement", "topic": "Share purchase terms"},
        {"name": "EmailThread_IPR_Dispute.msg", "type": "Correspondence", "topic": "IPR ownership issue"},
        {"name": "LegalNotice_BreachWarranty.docx", "type": "Summons", "topic": "Breach of contract"},
        {"name": "Statement_CA_Witness.pdf", "type": "Evidence", "topic": "Financial disclosures"},
        {"name": "SettlementProposal_Reliance.msg", "type": "Correspondence", "topic": "Out-of-court settlement"},
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
