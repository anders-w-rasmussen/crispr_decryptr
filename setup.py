import setuptools
  

setuptools.setup(
    name="crispr_decryptr",
    version="1.0.0",
    author="Anders Rasmussen",
    author_email="arasmussen@flatironinstitute.org",
    description="A statistical method for the analysis of CRISPR noncoding screens",
    url="https://github.com/anders-w-rasmussen/crispr_decryptr",
    packages=['decryptr/classify', 'decryptr/conv_mat_construct', 'decryptr/mcmc_glm'],
    package_data={"": ["*.tsv", "*.txt", "*.cpp", "*.h5", "*.stan", "bigWigToWig"],
    },
classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=['bin/decryptr'],
    include_package_data=True,
  
    install_requires=[
                        'matplotlib',
                        'numpy',
                        'keras',
                        'bioseq',
                        'biopython',
                        'alive_progress',
                        'pandas',
                        'tqdm',
                        'halo',
                        'cmdstanpy',
                        'tensorflow',
                        ],
  
  
  
  
  
)
