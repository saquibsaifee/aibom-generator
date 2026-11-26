import unittest
from unittest.mock import MagicMock, patch
from cyclonedx.model import ExternalReferenceType
from src.models.service import AIBOMService

class TestService(unittest.TestCase):
    def setUp(self):
        self.service = AIBOMService(hf_token="fake_token")
        self.service.hf_api = MagicMock()
        
    def test_normalise_model_id(self):
        self.assertEqual(AIBOMService._normalise_model_id("owner/model"), "owner/model")
        self.assertEqual(AIBOMService._normalise_model_id("https://huggingface.co/owner/model"), "owner/model")
        self.assertEqual(AIBOMService._normalise_model_id("https://huggingface.co/owner/model/tree/main"), "owner/model")

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_generate_aibom_basic(self, mock_extractor_cls, mock_score):
        # Mock dependencies
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "author": "tester"}
        mock_extractor.extraction_results = {}
        
        mock_score.return_value = {"total_score": 50}

        self.service.hf_api = MagicMock()
        self.service.hf_api.model_info.return_value = MagicMock(sha="123456")
        self.service.hf_api.model_card.return_value = MagicMock(data=MagicMock(to_dict=lambda: {}))
        
        aibom = self.service.generate_aibom("owner/test-model")
        
        self.assertIsNotNone(aibom)
        # Metadata component name is timestamp by default, check ML component instead
        self.assertEqual(aibom["components"][0]["name"], "test-model")
        self.assertEqual(aibom["bomFormat"], "CycloneDX")

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_generate_aibom_purl_encoding(self, mock_extractor_cls, mock_score):
        # Setup
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "author": "tester", "commit": "123456"}
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}

        self.service.hf_api = MagicMock()
        self.service.hf_api.model_info.return_value = MagicMock(sha="123456")
        
        # Action
        model_id = "owner/model"
        aibom = self.service.generate_aibom(model_id)
        
        # Verify PURL encoding (slash should be / now, case preserved)
        # Expected: pkg:huggingface/owner/model@123456
        ml_cmp = aibom["components"][0]
        
        self.assertEqual(ml_cmp["version"], "123456")
        self.assertIn("pkg:huggingface/owner/model@123456", ml_cmp["purl"])
        self.assertIn("pkg:huggingface/owner/model@123456", ml_cmp["bom-ref"])
        
    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_generate_aibom_version_truncation(self, mock_extractor_cls, mock_score):
        # Setup
        mock_extractor = mock_extractor_cls.return_value
        long_sha = "a" * 40
        # Extractor typically puts commit in metadata if available
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "commit": long_sha}
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}
        
        self.service.hf_api.model_info.return_value = MagicMock(sha=long_sha)
        
        # Action
        aibom = self.service.generate_aibom("owner/model")
        
        # Verify
        ml_cmp = aibom["components"][0]
        expected_version = "aaaaaaaa" # First 8 chars
        
        self.assertEqual(ml_cmp["version"], expected_version)
        self.assertIn(f"@{expected_version}", ml_cmp["purl"])
        self.assertIn(f"@{expected_version}", ml_cmp["bom-ref"])
        
        # Verify dependency graph: metadata component (ref) → dependsOn → model component
        # Only the dependsOn entry contains the model's versioned PURL
        self.assertIn(f"@{expected_version}", aibom["dependencies"][0]["dependsOn"][0])

    def test_infer_io_formats(self):
        # Test Text Classification
        inputs, outputs = self.service._infer_io_formats("text-classification")
        self.assertEqual(inputs, ["string"])
        self.assertEqual(outputs, ["string"])
        
        # Test Image Classification
        inputs, outputs = self.service._infer_io_formats("image-classification")
        self.assertEqual(inputs, ["image"])
        self.assertEqual(outputs, ["string", "json"])
        
        # Test ASR (Audio)
        inputs, outputs = self.service._infer_io_formats("automatic-speech-recognition")
        self.assertEqual(inputs, ["audio"])
        self.assertEqual(outputs, ["string"])
        
        # Test Unknown
        inputs, outputs = self.service._infer_io_formats("unknown-task")
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [])

    def test_create_aibom_structure_uses_cyclonedx_outputter(self):
        metadata = {
            "name": "test-model",
            "author": "tester",
            "commit": "1234567890abcdef"
        }

        aibom = self.service._create_aibom_structure("owner/test-model", metadata)

        self.assertEqual(aibom["bomFormat"], "CycloneDX")
        self.assertEqual(aibom["specVersion"], "1.6")
        self.assertIn("$schema", aibom)
        self.assertEqual(aibom["components"][0]["type"], "machine-learning-model")

    def test_create_aibom_structure_uses_valid_ethical_considerations(self):
        metadata = {
            "name": "test-model",
            "ethicalConsiderations": "May reflect bias in training data."
        }

        aibom = self.service._create_aibom_structure("owner/test-model", metadata)

        ethical_considerations = (
            aibom["components"][0]["modelCard"]["considerations"]["ethicalConsiderations"]
        )
        self.assertEqual(
            ethical_considerations,
            [{"name": "May reflect bias in training data."}]
        )
        self.assertNotIn("description", ethical_considerations[0])

    def test_create_minimal_aibom(self):
        aibom = self.service._create_minimal_aibom("owner/model")
        self.assertEqual(aibom["bomFormat"], "CycloneDX")
        self.assertEqual(aibom["specVersion"], "1.6")
        self.assertEqual(aibom["components"][0]["type"], "machine-learning-model")

    def test_external_reference_type_mapping_defaults_to_website(self):
        self.assertEqual(
            self.service._map_external_reference_type("documentation"),
            ExternalReferenceType.DOCUMENTATION
        )
        self.assertEqual(
            self.service._map_external_reference_type("totally-unknown-type"),
            ExternalReferenceType.WEBSITE
        )

    def test_build_cyclonedx_component_populates_native_model_card(self):
        from cyclonedx.model.model_card import ModelCard as CdxModelCard

        metadata = {
            "name": "test-model",
            "commit": "1234567890abcdef",
            "primaryPurpose": "text-classification",
            "typeOfModel": "bert",
            "datasets": ["glue"],
            "eval_results": [{"metric_type": "accuracy", "metric_value": 0.94}],
            "intendedUse": "Classify customer support tickets.",
            "technicalLimitations": "Limited to 512 tokens.",
            "ethicalConsiderations": "May reflect training-data bias.",
            "vocab_size": 30522,
        }

        with patch.object(self.service, "_verify_dataset_exists_on_hf", return_value=True):
            component_section = self.service._create_component_section("owner/test-model", metadata)

        component = self.service._build_cyclonedx_component(component_section)
        self.assertIsInstance(component.model_card, CdxModelCard)
        self.assertIsNotNone(component.model_card.model_parameters)
        self.assertEqual(component.model_card.model_parameters.task, "text-classification")
        self.assertEqual(component.model_card.model_parameters.model_architecture, "bert")
        self.assertEqual([i.format for i in component.model_card.model_parameters.inputs], ["string"])
        self.assertEqual([o.format for o in component.model_card.model_parameters.outputs], ["string"])

        self.assertIsNotNone(component.model_card.quantitative_analysis)
        metrics = list(component.model_card.quantitative_analysis.performance_metrics)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].type, "accuracy")
        self.assertEqual(metrics[0].value, "0.94")

        self.assertIsNotNone(component.model_card.considerations)
        self.assertIn("Classify customer support tickets.", component.model_card.considerations.use_cases)
        self.assertIn("Limited to 512 tokens.", component.model_card.considerations.technical_limitations)
        ethical = list(component.model_card.considerations.ethical_considerations)
        self.assertEqual(len(ethical), 1)
        self.assertEqual(ethical[0].name, "May reflect training-data bias.")

        prop_map = {p.name: p.value for p in component.model_card.properties}
        self.assertEqual(prop_map.get("genai:aibom:modelcard:vocabSize"), "30522")
        self.assertEqual(prop_map.get("genai:aibom:trainingDataAvailable"), "true")

    def test_create_aibom_structure_serializes_model_card_and_datasets_valid_schema(self):
        from src.utils.validation import validate_aibom

        metadata = {
            "name": "test-model",
            "author": "tester",
            "commit": "1234567890abcdef",
            "primaryPurpose": "text-classification",
            "typeOfModel": "bert",
            "datasets": ["squad", "glue"],
            "eval_results": [{"metric_type": "f1", "metric_value": "0.91"}],
            "intendedUse": "Sentiment classification",
            "technicalLimitations": "English only",
            "ethicalConsiderations": "May reflect training-data bias.",
            "vocab_size": 30522,
        }

        with patch.object(self.service, "_verify_dataset_exists_on_hf", return_value=True):
            aibom = self.service._create_aibom_structure("owner/test-model", metadata)

        model_card = aibom["components"][0]["modelCard"]
        params = model_card["modelParameters"]
        self.assertEqual(params["task"], "text-classification")
        self.assertEqual(params["modelArchitecture"], "bert")
        self.assertEqual(params["inputs"], [{"format": "string"}])
        self.assertEqual(params["outputs"], [{"format": "string"}])
        self.assertEqual(
            [d["name"] for d in params["datasets"]],
            ["squad", "glue"],
        )
        self.assertEqual(
            model_card["quantitativeAnalysis"]["performanceMetrics"],
            [{"type": "f1", "value": "0.91"}],
        )
        self.assertEqual(model_card["considerations"]["useCases"], ["Sentiment classification"])
        self.assertEqual(model_card["considerations"]["technicalLimitations"], ["English only"])
        self.assertEqual(
            model_card["considerations"]["ethicalConsiderations"],
            [{"name": "May reflect training-data bias."}],
        )

        is_valid, errors = validate_aibom(aibom)
        self.assertTrue(is_valid, f"Schema validation errors: {errors}")

    def test_model_card_from_dict_empty_and_datasets_only(self):
        self.assertIsNone(self.service._model_card_from_dict(None))
        self.assertIsNone(self.service._model_card_from_dict({}))
        # Datasets-only dict must not raise NotImplementedError from cyclonedx-python-lib
        self.assertIsNone(
            self.service._model_card_from_dict(
                {"modelParameters": {"datasets": [{"type": "dataset", "name": "ds"}]}}
            )
        )

    def test_model_card_from_dict_flattens_nested_consideration_lists(self):
        cdx_card = self.service._model_card_from_dict(
            {
                "considerations": {
                    "technicalLimitations": [["limit a", "limit b"]],
                    "useCases": ["use case 1"],
                }
            }
        )
        self.assertIsNotNone(cdx_card)
        self.assertEqual(
            list(cdx_card.considerations.technical_limitations),
            ["limit a", "limit b"],
        )
        self.assertEqual(list(cdx_card.considerations.use_cases), ["use case 1"])


class TestProcessLicenses(unittest.TestCase):
    def setUp(self):
        self.service = AIBOMService(hf_token="fake_token")

    # --- empty / missing inputs ---
    def test_no_license_returns_empty(self):
        self.assertEqual(self.service._process_licenses({}), [])

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.service._process_licenses({"license": ""}), [])

    def test_empty_list_returns_empty(self):
        self.assertEqual(self.service._process_licenses({"license": []}), [])

    # --- placeholder / generic values are skipped ---
    def test_noassertion_skipped(self):
        self.assertEqual(self.service._process_licenses({"license": "NOASSERTION"}), [])

    def test_other_skipped(self):
        self.assertEqual(self.service._process_licenses({"license": "other"}), [])

    def test_unknown_skipped(self):
        self.assertEqual(self.service._process_licenses({"license": "unknown"}), [])

    def test_none_value_skipped(self):
        self.assertEqual(self.service._process_licenses({"license": "none"}), [])

    # --- valid SPDX simple IDs ---
    def test_valid_spdx_id_produces_id_field(self):
        result = self.service._process_licenses({"license": "MIT"})
        self.assertEqual(result, [{"license": {"id": "MIT"}}])

    def test_lowercase_spdx_id_normalized(self):
        result = self.service._process_licenses({"license": "apache-2.0"})
        self.assertEqual(result, [{"license": {"id": "Apache-2.0"}}])

    def test_multi_word_alias_normalized(self):
        result = self.service._process_licenses({"license": "apache license 2.0"})
        self.assertEqual(result, [{"license": {"id": "Apache-2.0"}}])

    # --- list input: first element used ---
    def test_list_license_uses_first_element(self):
        result = self.service._process_licenses({"license": ["MIT", "Apache-2.0"]})
        self.assertEqual(result, [{"license": {"id": "MIT"}}])

    # --- licenses key takes precedence over license ---
    def test_licenses_key_used(self):
        result = self.service._process_licenses({"licenses": "MIT"})
        self.assertEqual(result, [{"license": {"id": "MIT"}}])

    # --- compound SPDX expression ---
    def test_compound_expression_produces_expression_field(self):
        result = self.service._process_licenses({"license": "MIT AND Apache-2.0"})
        self.assertEqual(len(result), 1)
        self.assertIn("expression", result[0])
        self.assertEqual(result[0]["expression"], "MIT AND Apache-2.0")

    # --- custom / non-SPDX name ---
    def test_custom_name_produces_name_field(self):
        result = self.service._process_licenses({"license": "nvidia-open-model-license"})
        self.assertEqual(len(result), 1)
        self.assertIn("license", result[0])
        lic = result[0]["license"]
        # nvidia-open-model-license is not a valid SPDX id → name field
        self.assertIn("name", lic)

    def test_custom_name_with_known_url_includes_url(self):
        result = self.service._process_licenses({"license": "nvidia open model license agreement"})
        self.assertEqual(len(result), 1)
        lic = result[0]["license"]
        self.assertIn("url", lic)
        self.assertIn("nvidia.com", lic["url"])


    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_training_data_flag_with_datasets(self, mock_extractor_cls, mock_score):
        """Test that trainingDataAvailable flag is set to true when datasets are present"""
        # Setup
        mock_extractor = mock_extractor_cls.return_value
        metadata_with_data = {
            "name": "test-model",
            "datasets": ["dataset1", "dataset2"],
            "commit": "123456"
        }
        mock_extractor.extract_metadata.return_value = metadata_with_data
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}

        self.service.hf_api = MagicMock()
        self.service.hf_api.model_info.return_value = MagicMock(sha="123456")
        
        # Mock dataset verification
        with patch.object(self.service, '_verify_dataset_exists_on_hf', return_value=True):
            # Action
            aibom = self.service.generate_aibom("owner/model")
        
        # Verify
        model_card = aibom["components"][0].get("modelCard", {})
        properties = model_card.get("properties", [])
        
        # Find the trainingDataAvailable property
        training_flag = next((p for p in properties if p["name"] == "genai:aibom:trainingDataAvailable"), None)
        self.assertIsNotNone(training_flag)
        self.assertEqual(training_flag["value"], "true")
        
        # Verify no warning
        warning = next((p for p in properties if p["name"] == "genai:aibom:trainingDataWarning"), None)
        self.assertIsNone(warning)

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_training_data_flag_without_datasets(self, mock_extractor_cls, mock_score):
        """Test that trainingDataAvailable flag is set to false and warning is added when datasets are missing"""
        # Setup
        mock_extractor = mock_extractor_cls.return_value
        metadata_no_data = {
            "name": "test-model",
            "commit": "123456"
            # No datasets key
        }
        mock_extractor.extract_metadata.return_value = metadata_no_data
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}

        self.service.hf_api = MagicMock()
        self.service.hf_api.model_info.return_value = MagicMock(sha="123456")
        
        # Action
        aibom = self.service.generate_aibom("owner/model")
        
        # Verify
        model_card = aibom["components"][0].get("modelCard", {})
        properties = model_card.get("properties", [])
        
        # Find the trainingDataAvailable property
        training_flag = next((p for p in properties if p["name"] == "genai:aibom:trainingDataAvailable"), None)
        self.assertIsNotNone(training_flag)
        self.assertEqual(training_flag["value"], "false")
        
        # Verify warning is present
        warning = next((p for p in properties if p["name"] == "genai:aibom:trainingDataWarning"), None)
        self.assertIsNotNone(warning)
        self.assertIn("Training data information is missing", warning["value"])

    def test_verify_datasets_available_with_valid_datasets(self):
        """Test dataset verification with valid datasets"""
        # Mock the HF API call
        with patch.object(self.service, '_verify_dataset_exists_on_hf', return_value=True):
            # List of valid datasets
            metadata = {"datasets": ["dataset1", "dataset2"]}
            self.assertTrue(self.service._verify_datasets_available(metadata))
            
            # Single string dataset
            metadata = {"datasets": "valid_dataset"}
            self.assertTrue(self.service._verify_datasets_available(metadata))
            
            # Dict format with name
            metadata = {"datasets": {"name": "my_dataset", "url": "https://example.com"}}
            self.assertTrue(self.service._verify_datasets_available(metadata))

    def test_verify_datasets_available_with_empty_datasets(self):
        """Test dataset verification with empty or invalid datasets"""
        # Empty list
        metadata = {"datasets": []}
        self.assertFalse(self.service._verify_datasets_available(metadata))
        
        # List with empty strings
        metadata = {"datasets": ["", "  ", ""]}
        self.assertFalse(self.service._verify_datasets_available(metadata))
        
        # Unknown placeholder
        metadata = {"datasets": ["unknown"]}
        self.assertFalse(self.service._verify_datasets_available(metadata))
        
        # No datasets key
        metadata = {"name": "test-model"}
        self.assertFalse(self.service._verify_datasets_available(metadata))

if __name__ == '__main__':
    unittest.main()
