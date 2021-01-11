# Frustration Intensity Prediction in Customer Support Dialog Texts

This repository contains Python 3 dialogue frustration intensity prediction source code.

## Instruction to run the program
    
  - Dialog input file should exist in the current directory named "dprocessed.txt" (example of a file given in the repository)
  - The parameters should be set as global variable CONFIG_LIST each element of which is pair of config items (see releated research paper "Frustration Intensity Prediction in Customer Support Dialog Texts"):
  -- USER_KEYWORD_COUNT
  -- SUPPORT_KEYWORD_COUNT
  - Optional parameters as localal variables can be also set:
  -- HIDDEN_COUNT
  -- EPOCHS
  - python3 file to be run without parameters
  - Output is written into "output.txt"
  - Details are written into "detail-" files
	
## Acknowledgements

The research has been supported by the European Regional Development Fund within the project “Multilingual Artificial Intelligence Based Human Computer Interaction” No. 1.1.1.1/18/A/148. 

## Releated research papers

Frustration Intensity Prediction in Customer Support Dialog Texts / Janis Zuters, Viktorija Leonova // 9th International Conference on Information Technology Convergence and Services (ITCS 2020), in David C. Wyld et al. (Eds): NLP, JSE, MLTEC, DMS, NeTIOT, ITCS, SIP, CST, ARIA – 2020 pp. 261-273, 2020. CS & IT - CSCP 2020. DOI: 10.5121/csit.2020.101419.

J. Zuters, V. Leonova. Adaptive Vocabulary Construction for Frustration Intensity Modelling in Customer Support Dialog Texts, International Journal of Computer Science and Information Technology (IJCSIT), December 2020, Volume 12, Number 6. http://aircconline.com/ijcsit/V12N6/12620ijcsit01.pdf


	