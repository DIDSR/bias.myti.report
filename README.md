# myti.report
A visualization tool to facilitate the comparison of user-implemented bias mitigation methods for AI models

# Documentation Questions
_Note: Remove this section prior to release._

1. **How can I view the documentation?** Download the docs folder and open index.html
2. **Why is the documentation not on GitHub Pages?** GitHub pages are always public. Until we are ready for public release, we'll have to download the docs folder to view the html.
3. **How do I update the documentation?** You can edit the text of the documentation directly in [docs/source/index.rst](docs/source/index.rst). Then regenerate the documentation using the command ``make html`` from within the docs folder on OpenHPC.
4. **Why is the documentation for a function/class not showing up?** Currently, index.rst has the command to automatically document the src module. Make sure that the file with your function/class is being imported into the src module in [src/\_\_init\_\_.py](src/__init__.py). If the function still doesn't show up, check that it has a docstring. 

