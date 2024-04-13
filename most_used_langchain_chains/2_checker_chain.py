from langchain.chains.llm_checker.base import LLMCheckerChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.5)
checker_chain = LLMCheckerChain.from_llm(llm=llm)

letter = """
To Whom It May Concern,

I am delighted to recommend Harrison Chase for opportunities in the field of machine learning. 
Harrison has been an outstanding Machine Learning Developer during his tenure at Kensho, and I am confident in his abilities and potential.

During his time at Kensho, Harrison made significant contributions to our machine learning projects. 
His expertise in developing and deploying machine learning models was instrumental in advancing several key initiatives. Harrison has a comprehensive understanding of various machine learning algorithms and frameworks, which he leveraged effectively to solve complex problems and optimize processes.

Harrison is distinguished by his analytical mindset and strong problem-solving skills. 
He consistently delivered high-quality solutions within challenging timelines, showcasing his ability to thrive under pressure. 
His proficiency in Python programming and experience with libraries like TensorFlow and PyTorch enabled him to implement scalable and efficient machine learning solutions.

Beyond his technical abilities, Harrison is a valuable team player with excellent communication skills. 
He actively collaborates with colleagues, seeks input, and contributes innovative ideas to drive projects forward. 
His leadership qualities and proactive approach were evident in his role at Kensho, where he took initiative and demonstrated a strong sense of ownership.

I am confident that Harrison's skills, work ethic, and collaborative nature make him a strong candidate for roles in machine learning development. 
He has a proven track record of success and would be an asset to any team seeking a dedicated and skilled machine learning professional.
"""

res = checker_chain.invoke(letter)

print(res)