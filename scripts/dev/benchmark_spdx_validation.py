import time
import timeit
import logging
from src.controllers.cli_controller import CLIController

logging.basicConfig(level=logging.WARNING)

valid_spdx_data = {
    "spdxVersion": "SPDX-2.3",
    "dataLicense": "CC0-1.0",
    "SPDXID": "SPDXRef-DOCUMENT",
    "name": "Test-Document",
    "documentNamespace": "http://spdx.org/spdxdocs/spdx-example-444504E0-4F89-41D3-9A0C-0305E82C3301",
    "creationInfo": {
        "creators": ["Tool: owasp-aibom-generator"],
        "created": "2023-11-20T14:30:00Z"
    },
    "packages": [
        {
            "name": "Test-Package",
            "SPDXID": "SPDXRef-Package",
            "downloadLocation": "NOASSERTION",
            "licenseDeclared": "NOASSERTION"
        }
    ],
    "relationships": [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relatedSpdxElement": "SPDXRef-Package",
            "relationshipType": "DESCRIBES"
        }
    ]
}

def benchmark_validation():
    controller = CLIController()

    # Run a single validation first to prove it works
    start = time.perf_counter()
    result = controller._validate_spdx_schema_version(valid_spdx_data, "2.3")
    end = time.perf_counter()
    print(f"Single validation result: {'SUCCESS' if result else 'FAILED'}")
    print(f"Time for single validation: {end - start:.4f} seconds\n")

    # Run multiple iterations using timeit
    iterations = 100
    print(f"Benchmarking with {iterations} iterations...")

    def validate():
        controller._validate_spdx_schema_version(valid_spdx_data, "2.3")

    total_time = timeit.timeit(validate, number=iterations)
    avg_time = total_time / iterations

    print(f"Total time for {iterations} iterations: {total_time:.4f} seconds")
    print(f"Average time per validation: {avg_time * 1000:.2f} milliseconds")

if __name__ == "__main__":
    benchmark_validation()
