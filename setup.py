from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="MonteCarloBarrierOptionPricer",
    version="0.1.0",
    author="Your Name", # Replace with your name
    author_email="your.email@example.com", # Replace with your email
    description="A Monte Carlo pricer for barrier options using Geometric Brownian Motion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/monte-carlo-barrier-option-pricer", # Replace with your repo URL
    packages=find_packages(exclude=("tests*", "notebooks*")), 
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha", # Or "4 - Beta", "5 - Production/Stable"
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8', # Specify your minimum Python version
    keywords='monte carlo, option pricing, barrier options, quantitative finance, gbm, derivatives',
    project_urls={ # Optional
        'Bug Reports': 'https://github.com/yourusername/monte-carlo-barrier-option-pricer/issues',
        'Source': 'https://github.com/yourusername/monte-carlo-barrier-option-pricer/',
    },
    # If you have command-line scripts defined in your package:
    # entry_points={
    #     'console_scripts': [
    #         'price_barrier_option=barrier_option_pricer.cli:main_cli_function', 
    #     ],
    # },
)
