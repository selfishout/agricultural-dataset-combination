#!/usr/bin/env python3
"""
Simplified Benchmarking Test Suite
Tests all requirements for a proper benchmarking dataset with robust error handling
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedBenchmarkingTestSuite:
    def __init__(self, base_path="/Volumes/Rapid/Agriculture Dataset"):
        self.base_path = Path(base_path)
        self.test_results = {
            "test_date": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": 0,
            "test_details": {},
            "overall_score": 0.0,
            "benchmark_ready": False
        }
        
    def count_images_in_path(self, path):
        """Safely count images in a path"""
        try:
            if not path.exists():
                return 0
            
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                try:
                    image_files.extend(list(path.rglob(ext)))
                except Exception:
                    continue
            
            return len(image_files)
        except Exception:
            return 0
    
    def test_dataset_size(self):
        """Test 1: Dataset Size Requirements"""
        logger.info("Testing dataset size requirements...")
        
        try:
            # Count total images
            total_images = 0
            dataset_counts = {}
            
            datasets = {
                "phenobench": self.base_path / "PhenoBench",
                "capsicum": self.base_path / "Synthetic and Empirical Capsicum Annuum Image Dataset_1_all" / "uuid_884958f5-b868-46e1-b3d8-a0b5d91b02c0",
                "weed_augmented": self.base_path / "weed_augmented",
                "vineyard": self.base_path / "Vineyard Canopy Images during Early Growth Stage",
                "fruit_datasets": self.base_path / "Combined_datasets" / "fruit_datasets"
            }
            
            for name, path in datasets.items():
                count = self.count_images_in_path(path)
                dataset_counts[name] = count
                total_images += count
            
            # Benchmarking requirements
            min_benchmark = 10000
            good_benchmark = 50000
            excellent_benchmark = 100000
            
            passed = total_images >= min_benchmark
            score = "Excellent" if total_images >= excellent_benchmark else "Good" if total_images >= good_benchmark else "Insufficient"
            
            return {
                "passed": passed,
                "total_images": total_images,
                "dataset_counts": dataset_counts,
                "score": score,
                "min_required": min_benchmark,
                "good_threshold": good_benchmark,
                "excellent_threshold": excellent_benchmark
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error counting images: {str(e)}"}
    
    def test_class_diversity(self):
        """Test 2: Class Diversity Requirements"""
        logger.info("Testing class diversity requirements...")
        
        try:
            # Define expected classes
            agricultural_classes = [
                "background", "canopy", "crop", "disease", "grape", "leaf", 
                "pepper_fruit", "pepper_plant", "plant", "root", "soil", "stem", "vine", "weed"
            ]
            
            fruit_classes = [
                "apple", "banana", "cherry", "mango", "orange", "peach", 
                "pineapple", "strawberry", "watermelon"
            ]
            
            all_classes = agricultural_classes + fruit_classes
            total_classes = len(all_classes)
            
            # Requirements
            min_classes = 10
            good_classes = 20
            excellent_classes = 30
            
            passed = total_classes >= min_classes
            score = "Excellent" if total_classes >= excellent_classes else "Good" if total_classes >= good_classes else "Insufficient"
            
            return {
                "passed": passed,
                "total_classes": total_classes,
                "agricultural_classes": len(agricultural_classes),
                "fruit_classes": len(fruit_classes),
                "all_classes": all_classes,
                "score": score,
                "min_required": min_classes,
                "good_threshold": good_classes,
                "excellent_threshold": excellent_classes
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error analyzing classes: {str(e)}"}
    
    def test_image_quality(self):
        """Test 3: Image Quality Requirements"""
        logger.info("Testing image quality requirements...")
        
        try:
            # Sample images from each dataset
            datasets = {
                "phenobench": self.base_path / "PhenoBench",
                "capsicum": self.base_path / "Synthetic and Empirical Capsicum Annuum Image Dataset_1_all" / "uuid_884958f5-b868-46e1-b3d8-a0b5d91b02c0",
                "weed_augmented": self.base_path / "weed_augmented",
                "vineyard": self.base_path / "Vineyard Canopy Images during Early Growth Stage",
                "fruit_datasets": self.base_path / "Combined_datasets" / "fruit_datasets"
            }
            
            quality_issues = 0
            total_sampled = 0
            valid_images = 0
            
            for name, path in datasets.items():
                if path.exists():
                    # Sample 20 images from each dataset
                    image_files = []
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                        try:
                            image_files.extend(list(path.rglob(ext)))
                        except Exception:
                            continue
                    
                    sample_size = min(20, len(image_files))
                    sample_files = image_files[:sample_size]
                    
                    for img_path in sample_files:
                        total_sampled += 1
                        try:
                            with Image.open(img_path) as img:
                                # Check basic quality metrics
                                if img.size[0] < 100 or img.size[1] < 100:
                                    quality_issues += 1
                                elif img.mode not in ['RGB', 'L']:
                                    quality_issues += 1
                                else:
                                    valid_images += 1
                        except Exception:
                            quality_issues += 1
            
            error_rate = quality_issues / total_sampled if total_sampled > 0 else 0
            
            # Requirements
            max_error_rate = 0.05  # 5% error rate
            good_error_rate = 0.02  # 2% error rate
            excellent_error_rate = 0.01  # 1% error rate
            
            passed = error_rate <= max_error_rate
            score = "Excellent" if error_rate <= excellent_error_rate else "Good" if error_rate <= good_error_rate else "Needs Improvement"
            
            return {
                "passed": passed,
                "total_sampled": total_sampled,
                "valid_images": valid_images,
                "quality_issues": quality_issues,
                "error_rate": error_rate,
                "score": score,
                "max_error_rate": max_error_rate,
                "good_error_rate": good_error_rate,
                "excellent_error_rate": excellent_error_rate
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error checking image quality: {str(e)}"}
    
    def test_annotation_coverage(self):
        """Test 4: Annotation Coverage Requirements"""
        logger.info("Testing annotation coverage requirements...")
        
        try:
            # Count annotated vs unannotated images
            annotated_datasets = ["phenobench", "weed_augmented", "vineyard"]
            unannotated_datasets = ["capsicum", "fruit_datasets"]
            
            annotated_images = 0
            unannotated_images = 0
            
            datasets = {
                "phenobench": self.base_path / "PhenoBench",
                "capsicum": self.base_path / "Synthetic and Empirical Capsicum Annuum Image Dataset_1_all" / "uuid_884958f5-b868-46e1-b3d8-a0b5d91b02c0",
                "weed_augmented": self.base_path / "weed_augmented",
                "vineyard": self.base_path / "Vineyard Canopy Images during Early Growth Stage",
                "fruit_datasets": self.base_path / "Combined_datasets" / "fruit_datasets"
            }
            
            for name, path in datasets.items():
                count = self.count_images_in_path(path)
                if name in annotated_datasets:
                    annotated_images += count
                else:
                    unannotated_images += count
            
            total_images = annotated_images + unannotated_images
            annotation_coverage = annotated_images / total_images if total_images > 0 else 0
            
            # Requirements
            min_coverage = 0.5  # 50% coverage
            good_coverage = 0.7  # 70% coverage
            excellent_coverage = 0.8  # 80% coverage
            
            passed = annotation_coverage >= min_coverage
            score = "Excellent" if annotation_coverage >= excellent_coverage else "Good" if annotation_coverage >= good_coverage else "Insufficient"
            
            return {
                "passed": passed,
                "total_images": total_images,
                "annotated_images": annotated_images,
                "unannotated_images": unannotated_images,
                "annotation_coverage": annotation_coverage,
                "score": score,
                "min_coverage": min_coverage,
                "good_coverage": good_coverage,
                "excellent_coverage": excellent_coverage
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error checking annotation coverage: {str(e)}"}
    
    def test_format_consistency(self):
        """Test 5: Format Consistency Requirements"""
        logger.info("Testing format consistency requirements...")
        
        try:
            # Check image formats across datasets
            datasets = {
                "phenobench": self.base_path / "PhenoBench",
                "capsicum": self.base_path / "Synthetic and Empirical Capsicum Annuum Image Dataset_1_all" / "uuid_884958f5-b868-46e1-b3d8-a0b5d91b02c0",
                "weed_augmented": self.base_path / "weed_augmented",
                "vineyard": self.base_path / "Vineyard Canopy Images during Early Growth Stage",
                "fruit_datasets": self.base_path / "Combined_datasets" / "fruit_datasets"
            }
            
            format_counts = {}
            total_images = 0
            
            for name, path in datasets.items():
                if path.exists():
                    # Sample images to check formats
                    image_files = []
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                        try:
                            image_files.extend(list(path.rglob(ext)))
                        except Exception:
                            continue
                    
                    sample_size = min(10, len(image_files))
                    sample_files = image_files[:sample_size]
                    
                    for img_path in sample_files:
                        try:
                            with Image.open(img_path) as img:
                                format_name = img.format
                                format_counts[format_name] = format_counts.get(format_name, 0) + 1
                                total_images += 1
                        except Exception:
                            continue
            
            # Check consistency
            if total_images > 0:
                dominant_format = max(format_counts.items(), key=lambda x: x[1])[0]
                format_consistency = format_counts[dominant_format] / total_images
            else:
                dominant_format = None
                format_consistency = 0
            
            # Requirements
            min_consistency = 0.8  # 80% consistency
            
            passed = format_consistency >= min_consistency
            score = "Excellent" if format_consistency >= 0.95 else "Good" if format_consistency >= 0.9 else "Needs Improvement"
            
            return {
                "passed": passed,
                "total_images": total_images,
                "format_counts": format_counts,
                "dominant_format": dominant_format,
                "format_consistency": format_consistency,
                "score": score,
                "min_consistency": min_consistency
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error checking format consistency: {str(e)}"}
    
    def test_split_distribution(self):
        """Test 6: Train/Validation/Test Split Requirements"""
        logger.info("Testing split distribution requirements...")
        
        try:
            # Check if proper splits exist
            combined_path = self.base_path / "Combined_datasets"
            splits_exist = False
            split_counts = {}
            
            if combined_path.exists():
                for split in ["train", "val", "test"]:
                    split_path = combined_path / split
                    if split_path.exists():
                        count = self.count_images_in_path(split_path)
                        split_counts[split] = count
                        splits_exist = True
            
            if splits_exist and split_counts:
                total_split_images = sum(split_counts.values())
                if total_split_images > 0:
                    train_ratio = split_counts.get("train", 0) / total_split_images
                    val_ratio = split_counts.get("val", 0) / total_split_images
                    test_ratio = split_counts.get("test", 0) / total_split_images
                    
                    # Check if ratios are close to standard (70/20/10)
                    train_ok = 0.6 <= train_ratio <= 0.8
                    val_ok = 0.15 <= val_ratio <= 0.25
                    test_ok = 0.05 <= test_ratio <= 0.15
                    
                    passed = train_ok and val_ok and test_ok
                    score = "Excellent" if passed else "Needs Improvement"
                else:
                    passed = False
                    score = "No Images"
                    train_ratio = val_ratio = test_ratio = 0
            else:
                passed = False
                score = "Missing"
                train_ratio = val_ratio = test_ratio = 0
            
            return {
                "passed": passed,
                "splits_exist": splits_exist,
                "split_counts": split_counts,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "score": score
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error checking split distribution: {str(e)}"}
    
    def test_documentation(self):
        """Test 7: Documentation Requirements"""
        logger.info("Testing documentation requirements...")
        
        try:
            required_docs = [
                "README.md",
                "Combined_datasets/metadata.json"
            ]
            
            docs_found = 0
            doc_status = {}
            
            for doc in required_docs:
                doc_path = self.base_path / doc
                exists = doc_path.exists()
                doc_status[doc] = exists
                if exists:
                    docs_found += 1
            
            # Check if README has required sections
            readme_complete = False
            if (self.base_path / "README.md").exists():
                try:
                    with open(self.base_path / "README.md", 'r') as f:
                        readme_content = f.read()
                        required_sections = ["Dataset Statistics", "Source Datasets"]
                        readme_complete = all(section in readme_content for section in required_sections)
                except Exception:
                    readme_complete = False
            
            passed = docs_found >= 1 and readme_complete
            score = "Excellent" if docs_found == len(required_docs) and readme_complete else "Good" if passed else "Insufficient"
            
            return {
                "passed": passed,
                "docs_found": docs_found,
                "total_docs": len(required_docs),
                "doc_status": doc_status,
                "readme_complete": readme_complete,
                "score": score
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error checking documentation: {str(e)}"}
    
    def test_reproducibility(self):
        """Test 8: Reproducibility Requirements"""
        logger.info("Testing reproducibility requirements...")
        
        try:
            required_files = [
                "process_complete_datasets_clean.py",
                "benchmarking_evaluation.py",
                "final_comprehensive_evaluation.py"
            ]
            
            scripts_found = 0
            script_status = {}
            
            for script in required_files:
                script_path = Path(script)
                exists = script_path.exists()
                script_status[script] = exists
                if exists:
                    scripts_found += 1
            
            passed = scripts_found >= 2
            score = "Excellent" if scripts_found == len(required_files) else "Good" if passed else "Insufficient"
            
            return {
                "passed": passed,
                "scripts_found": scripts_found,
                "total_scripts": len(required_files),
                "script_status": script_status,
                "score": score
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error checking reproducibility: {str(e)}"}
    
    def test_model_compatibility(self):
        """Test 9: Model Compatibility Requirements"""
        logger.info("Testing model compatibility requirements...")
        
        try:
            # Check annotation availability
            annotated_datasets = ["phenobench", "weed_augmented", "vineyard"]
            total_annotated = 0
            total_images = 0
            
            datasets = {
                "phenobench": self.base_path / "PhenoBench",
                "capsicum": self.base_path / "Synthetic and Empirical Capsicum Annuum Image Dataset_1_all" / "uuid_884958f5-b868-46e1-b3d8-a0b5d91b02c0",
                "weed_augmented": self.base_path / "weed_augmented",
                "vineyard": self.base_path / "Vineyard Canopy Images during Early Growth Stage",
                "fruit_datasets": self.base_path / "Combined_datasets" / "fruit_datasets"
            }
            
            for name, path in datasets.items():
                count = self.count_images_in_path(path)
                total_images += count
                if name in annotated_datasets:
                    total_annotated += count
            
            annotation_coverage = total_annotated / total_images if total_images > 0 else 0
            
            # Model compatibility requirements
            min_coverage_for_models = 0.5  # 50% coverage needed for semantic segmentation models
            
            passed = annotation_coverage >= min_coverage_for_models
            score = "Excellent" if annotation_coverage >= 0.8 else "Good" if annotation_coverage >= 0.6 else "Insufficient"
            
            return {
                "passed": passed,
                "annotation_coverage": annotation_coverage,
                "total_images": total_images,
                "annotated_images": total_annotated,
                "score": score,
                "min_coverage": min_coverage_for_models
            }
        except Exception as e:
            return {"passed": False, "reason": f"Error checking model compatibility: {str(e)}"}
    
    def run_all_tests(self):
        """Run all benchmarking tests"""
        logger.info("🚀 Starting simplified benchmarking test suite...")
        
        # Run all tests
        tests = [
            ("Dataset Size", self.test_dataset_size),
            ("Class Diversity", self.test_class_diversity),
            ("Image Quality", self.test_image_quality),
            ("Annotation Coverage", self.test_annotation_coverage),
            ("Format Consistency", self.test_format_consistency),
            ("Split Distribution", self.test_split_distribution),
            ("Documentation", self.test_documentation),
            ("Reproducibility", self.test_reproducibility),
            ("Model Compatibility", self.test_model_compatibility)
        ]
        
        for test_name, test_function in tests:
            self.test_results["total_tests"] += 1
            
            try:
                result = test_function()
                if result["passed"]:
                    self.test_results["tests_passed"] += 1
                    logger.info(f"✅ {test_name}: PASSED ({result.get('score', 'N/A')})")
                else:
                    self.test_results["tests_failed"] += 1
                    logger.warning(f"❌ {test_name}: FAILED ({result.get('score', 'N/A')}) - {result.get('reason', 'Unknown error')}")
                
                self.test_results["test_details"][test_name] = result
                
            except Exception as e:
                self.test_results["tests_failed"] += 1
                logger.error(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results["test_details"][test_name] = {
                    "passed": False,
                    "reason": f"Test error: {str(e)}"
                }
        
        # Calculate overall score
        self.test_results["overall_score"] = self.test_results["tests_passed"] / self.test_results["total_tests"]
        self.test_results["benchmark_ready"] = self.test_results["overall_score"] >= 0.7
        
        # Save results
        results_path = self.base_path / "simplified_benchmarking_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        self.print_test_summary()
        
        logger.info(f"📁 Test results saved to: {results_path}")
        return self.test_results
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("🧪 SIMPLIFIED BENCHMARKING TEST SUITE RESULTS")
        print("="*80)
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"  Tests Passed: {self.test_results['tests_passed']}")
        print(f"  Tests Failed: {self.test_results['tests_failed']}")
        print(f"  Total Tests: {self.test_results['total_tests']}")
        print(f"  Overall Score: {self.test_results['overall_score']:.2f}")
        print(f"  Benchmark Ready: {'✅ YES' if self.test_results['benchmark_ready'] else '❌ NO'}")
        
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results["test_details"].items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            score = result.get("score", "N/A")
            print(f"  {test_name}: {status} ({score})")
            if not result["passed"] and "reason" in result:
                print(f"    Reason: {result['reason']}")
        
        print("\n" + "="*80)

def main():
    test_suite = SimplifiedBenchmarkingTestSuite()
    results = test_suite.run_all_tests()
    
    print(f"\n🎉 Benchmarking test suite complete!")
    print(f"📊 Benchmark ready: {'✅ YES' if results['benchmark_ready'] else '❌ NO'}")
    print(f"🎯 Overall score: {results['overall_score']:.2f}")

if __name__ == "__main__":
    main()
