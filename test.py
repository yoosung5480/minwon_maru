from minwon_maru.engine.chat import ChatManager, ProjPath


paths = ProjPath(
        data_root_str="/Users/yujin/Desktop/코딩shit/python_projects/대한민국해커톤/민원마루ver1/datas",
        metadata_name="metadata.json",
        raw_docs_dir="rawDocs",
        parsed_docs_dir="parsedDocs",
        workpages_name="work_page_info.json",
        reference_root_str="reference",
        departemt_info_name="department_info.json"
    )

manager = ChatManager(paths)
chat_id = manager.start_chat()


sample_user_q = "금정도서관 주차장 이용은 공짜야? 다음주 월요일에 방문하면 몇시까지해?"
sample_reps = manager.ask(sample_user_q)
print("user input : ", sample_user_q)
print("test output : ", sample_reps["generation"])