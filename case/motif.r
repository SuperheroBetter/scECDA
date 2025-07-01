library(Signac)
library(Seurat)
library(JASPAR2020)
library(TFBSTools)
library(motifmatchr)
library(BSgenome.Hsapiens.UCSC.hg38)
library(EnsDb.Hsapiens.v86)
library(chromVAR)
library(ggseqlogo)
library(SeuratDisk)
library(patchwork)
library(dplyr)
library(Signac)
library(GenomicRanges)
library(ggplot2)
library(tidyverse)
set.seed(1234)

setwd("E:/R_motif")

# pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
# pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz
# https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k

counts <- Read10X_h5('./pbmc10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5')
atac = counts$Peaks

chrom_assay <- CreateChromatinAssay(
  counts = atac,
  sep = c(":", "-"),
  genome = 'hg38',
  fragments = 'pbmc10k/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz',
  min.cells = 10,
  min.features = 200
)

pfm <- getMatrixSet(
  x = JASPAR2020,
  opts = list(collection = "CORE", 
              tax_group = 'vertebrates',
              all_versions = FALSE)
)

#### create Seurat object----
pbmc <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "Peaks",
  meta.data = metadata
)

# annotation
# extract gene annotations from EnsDb
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)

# change to UCSC style since the data was mapped to hg38
seqlevelsStyle(annotations) <- 'UCSC'

# add the gene information to the object
Annotation(pbmc) <- annotations
# head(Annotation(pbmc))

pbmc <- NucleosomeSignal(object = pbmc)

# compute TSS enrichment score per cell
pbmc <- TSSEnrichment(object = pbmc, fast = FALSE)

# Compute counts per cell in gene body and promoter region
gene.activities <- GeneActivity(pbmc)

# add motif information
pbmc <- AddMotifs(
  object = pbmc,
  genome = BSgenome.Hsapiens.UCSC.hg38,
  pfm = pfm
)

GetAssayData(object = pbmc, slot = "motifs")

features.keep <- as.character(seqnames(granges(pbmc))) %in% standardChromosomes(granges(pbmc))
pbmc <- pbmc[features.keep, ]# if you have multiple assays you'll need to adjust this to keep features from the different assays


pbmc <- RunChromVAR(
  object = pbmc,
  genome = BSgenome.Hsapiens.UCSC.hg38
)

#### Normalization ----
pbmc <- RunTFIDF(pbmc)

#### extract highly variable peaks----
pbmc <- FindTopFeatures(pbmc, min.cutoff = 50, n = 10000)
top1e4 <- head(VariableFeatures(pbmc), 10000)
pbmc <- subset(pbmc, features = top1e4)

#### Dimention reduction ----
pbmc <- RunSVD(pbmc)
DepthCor(pbmc)
pbmc <- RunUMAP(object = pbmc, reduction = 'lsi', dims = 2:30) 

#### Clustering ----
pbmc <- FindNeighbors(object = pbmc, reduction = 'lsi', dims = 2:30)
pbmc <- FindClusters(object = pbmc, verbose = FALSE, algorithm = 4, random.seed=6) 
p1 <- DimPlot(
  object = pbmc,
  group.by = "seurat_clusters",  
  label = TRUE,                   
  label.size = 4,                 
  repel = TRUE,                   
  reduction = "umap"              
) +
  ggtitle("PBMC Clusters")      

#### Marker genes ----
da_peaks <- Seurat::FindMarkers(
  object = pbmc,
  ident.1 = '12',
  ident.2 = '14',
  only.pos = TRUE,
  test.use = 'LR',
  min.pct = 0.05,
  latent.vars = 'nCount_Peaks'
)
filtered_da_peaks <- da_peaks[grepl("^chr", row.names(da_peaks)), ]
top.da.peak <- rownames(filtered_da_peaks[filtered_da_peaks$p_val < 0.005, ])

# test enrichment
enriched.motifs <- FindMotifs(
  object = pbmc,
  features = top.da.peak
)

MotifPlot(
 object = pbmc,
 motifs = head(rownames(enriched.motifs)),
 assay = 'Peaks'
)

DefaultAssay(pbmc) <- 'chromvar'
FeaturePlot(
 object = pbmc,
 features = head(rownames(enriched.motifs)),
 min.cutoff = 'q10',
 max.cutoff = 'q90',
 pt.size = 0.1
)

# differential motifs
DefaultAssay(pbmc) <- 'chromvar'
da_motifs <- Seurat::FindMarkers(
 object = pbmc,
 ident.1 = 'CD4 Naive',
 ident.2 = 'CD8 Naive',
 only.pos = TRUE,
 test.use = 'LR',
 min.pct = 0.05,
 latent.vars = 'nCount_peaks'
)
head(da_motifs)

# `10x.csv` from https://pan.baidu.com/s/1LfNgQrcX0S1iQIA8cNgBAA?pwd=jiad

# load `hidden` information
hidden <- read_csv('10x.csv')
pbmc_subset <- subset(pbmc, cells = hidden$cell_id)  
# cell_id alignment
hidden <- hidden[match(colnames(pbmc_subset), hidden$cell_id), ]
# test 
# identical(hidden$cell_id, colnames(pbmc_subset))  # return TRUE
umap_embeddings <- as.matrix(hidden[, c("umap1", "umap2")])
rownames(umap_embeddings) <- hidden$cell_id
colnames(umap_embeddings) <- c("UMAP_1", "UMAP_2")  # 必须包含"UMAP_"前缀

# create new DimReduc object（key step）
new_umap <- CreateDimReducObject(
  embeddings = umap_embeddings,
  key = "UMAP_",  # prefix must consistent with colnames(umap_embeddings)
  assay = "Peaks"   # specify correlated Assay
)

pbmc_subset[["umap"]] <- new_umap

# test
# head(pbmc_subset@reductions$umap@cell.embeddings)  
# DimPlot(pbmc_subset, reduction = "umap")           

# Add true label (cell_type)
pbmc_subset$cell_type <- hidden$cell_type

# Add predicted clusters  (cluster)
pbmc_subset$cluster <- hidden$cluster

umap_true_pl = DimPlot(pbmc_subset, group.by = "cell_type")  
umap_cluster_pl = DimPlot(pbmc_subset, group.by = "cluster")  

umap_true_pl <- lapply(umap_true_pl, function(p) {
  p + 
    theme(
      axis.line = element_blank(),        
      axis.text = element_blank(),        
      axis.ticks = element_blank(),       
      axis.title = element_blank()        
    ) 
})

umap_cluster_pl <- lapply(umap_cluster_pl, function(p) {
  p + 
    theme(
      axis.line = element_blank(),        
      axis.text = element_blank(),        
      axis.ticks = element_blank(),      
      axis.title = element_blank()      
    ) 
})

DefaultAssay(pbmc_subset) <- 'chromvar'  
Idents(pbmc_subset) <- "cluster"       

# compute differential motifs
all_motifs <- FindAllMarkers(
  object = pbmc_subset,
  only.pos = TRUE,        
  test.use = 'LR',        
  min.pct = 0.05,        
  latent.vars = 'nCount_Peaks', 
  logfc.threshold = 0.1
)

significant_motifs <- all_motifs %>%
  filter(p_val_adj < 0.05) %>%     
  group_by(cluster) %>%             
  arrange(p_val_adj, desc(avg_log2FC)) 

write.csv(significant_motifs, "all_celltypes_significant_motifs.csv")

top_motifs <- significant_motifs %>% group_by(cluster) %>% top_n(5, avg_log2FC)
DoHeatmap(
  pbmc_subset,
  features = top_motifs$gene,
  assay = "chromvar",
  slot = "data", 
  group.colors = scales::hue_pal()(nlevels(pbmc_subset))
)

plot_list <- FeaturePlot(
  object = pbmc_subset,
  features = c("MA0687.1", "MA0496.3", "MA0017.2", "IRF9",
               "GATA2", "MA1491.1"), 
  reduction = "umap",
  slot = "data",  
  combine = F,
  order = TRUE,
  min.cutoff = 'q05',  
  max.cutoff = 'q95'   
)

# combine figures
library(patchwork)
plot_list <- lapply(plot_list, function(p) {
  p + 
    theme(
      axis.line = element_blank(),       
      axis.text = element_blank(),       
      axis.ticks = element_blank(),     
      axis.title = element_blank()       
    ) 
})

combined_plot <- wrap_plots(plot_list, ncol = 2)
print(combined_plot)
