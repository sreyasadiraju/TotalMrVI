import torch
import torch.nn as nn # For nn.Dropout in dropout tests
from totalmrvi.decoders import DecoderZXAttention, ProteinDecoderZYAttention

# Test functions for DecoderZXAttention (from Cell 604 and new test)
def test_decoder_zx_attention_gene_dispersion():
    decoder=DecoderZXAttention(n_latent=16,n_output_genes=100,n_batch=4,dispersion="gene")
    z=torch.randn(8,16); batch_index=torch.randint(0,4,(8,)); library=torch.log(torch.rand(8,1)*100+1)
    px_scale,px_rate,px_r=decoder(z,batch_index,library)
    assert px_scale.shape==(8,100) and px_rate.shape==(8,100) and px_r.shape==(8,100)
    assert torch.allclose(px_scale.sum(dim=-1),torch.ones(8),atol=1e-4) and torch.all(px_r > 0)
    print("✅ test_decoder_zx_attention_gene_dispersion passed.")

def test_decoder_zx_attention_gene_cell_dispersion():
    decoder=DecoderZXAttention(n_latent=16,n_output_genes=100,n_batch=4,dispersion="gene-cell")
    z=torch.randn(8,16); batch_index=torch.randint(0,4,(8,)); library=torch.log(torch.rand(8,1)*100+1)
    px_scale,px_rate,px_r=decoder(z,batch_index,library)
    assert px_scale.shape==(8,100) and px_rate.shape==(8,100) and px_r.shape==(8,100)
    assert torch.allclose(px_scale.sum(dim=-1),torch.ones(8),atol=1e-4) and torch.all(px_r > 0)
    print("✅ test_decoder_zx_attention_gene_cell_dispersion passed.")

def test_decoder_zx_attention_variable_shapes():
    for batch_size in [1,5,32]:
        for n_genes in [10,50]:
            decoder=DecoderZXAttention(n_latent=16,n_output_genes=n_genes,n_batch=4,dispersion="gene")
            z=torch.randn(batch_size,16);batch_index=torch.randint(0,4,(batch_size,));library=torch.log(torch.rand(batch_size,1)*100+1)
            px_scale,px_rate,px_r=decoder(z,batch_index,library)
            assert px_scale.shape==(batch_size,n_genes) and px_rate.shape==(batch_size,n_genes) and px_r.shape==(batch_size,n_genes)
    print("✅ test_decoder_zx_attention_variable_shapes passed.")

def test_decoder_zx_attention_s_dimension():
    S,B,D_latent,N_batch,G=3,8,16,4,100
    decoder=DecoderZXAttention(n_latent=D_latent,n_output_genes=G,n_batch=N_batch,dispersion="gene-cell")
    z_sb=torch.randn(S,B,D_latent);batch_index_b=torch.randint(0,N_batch,(B,));library_sb=torch.log(torch.rand(S,B,1)*100+1)
    px_scale,px_rate,px_r=decoder(z_sb,batch_index_b,library_sb)
    assert px_scale.shape==(S,B,G) and px_rate.shape==(S,B,G) and px_r.shape==(S,B,G)
    print("✅ DecoderZXAttention S-dimension test passed!")

def test_decoder_zx_attention_batch_effect():
    B,D_latent,N_batch,G=8,16,4,100
    decoder=DecoderZXAttention(n_latent=D_latent,n_output_genes=G,n_batch=N_batch,dispersion="gene-cell")
    z=torch.randn(B,D_latent);batch_index=torch.randint(0,N_batch,(B,));library=torch.log(torch.rand(B,1)*100+1)
    _,px_rate1,_=decoder(z,batch_index,library);_,px_rate2,_=decoder(z,(batch_index+1)%N_batch,library)
    assert not torch.allclose(px_rate1,px_rate2)
    print("✅ DecoderZXAttention batch effect test passed!")

def test_decoder_zx_attention_dropout():
    torch.manual_seed(0)
    decoder=DecoderZXAttention(n_latent=16,n_output_genes=100,n_batch=4,dispersion="gene-cell",dropout_rate_attn=0.5)
    z=torch.randn(8,16);batch_index=torch.randint(0,4,(8,));library=torch.log(torch.rand(8,1)*100+1)
    decoder.eval();px_scale_eval,_,_=decoder(z,batch_index,library)
    decoder.train();px_scale_train1,_,_=decoder(z,batch_index,library);px_scale_train2,_,_=decoder(z,batch_index,library)
    assert not torch.allclose(px_scale_eval,px_scale_train1)
    assert not torch.allclose(px_scale_train1,px_scale_train2)
    has_dropout=any(isinstance(m,nn.Dropout) for m in decoder.residual_mlp.modules())
    assert has_dropout
    print("✅ DecoderZXAttention dropout test passed.")


# Test functions for ProteinDecoderZYAttention (from Cell 607 and new test)
def test_protein_decoder_shapes_and_values():
    B,D,P=10,32,5;z=torch.randn(B,D);logbeta=torch.randn(B,P);batch_index=torch.randint(0,3,(B,))
    decoder=ProteinDecoderZYAttention(n_latent=D,n_output_proteins=P,n_batch=3,dispersion="protein-cell")
    rate_back,rate_fore,mix_logit,_,py_r=decoder(z,batch_index,logbeta)
    assert rate_back.shape==(B,P) and rate_fore.shape==(B,P) and mix_logit.shape==(B,P) and py_r.shape==(B,P)
    assert torch.all(rate_back>0) and torch.all(rate_fore>rate_back)
    print("✅ test_protein_decoder_shapes_and_values passed!")

def test_protein_decoder_constant_dispersion():
    B,D,P=8,16,7;z=torch.randn(B,D);logbeta=torch.randn(B,P);batch_index=torch.randint(0,2,(B,))
    decoder=ProteinDecoderZYAttention(n_latent=D,n_output_proteins=P,n_batch=2,dispersion="protein")
    _,_,_,_,py_r=decoder(z,batch_index,logbeta)
    assert py_r.shape==(B,P) and torch.all(py_r>0)
    print("✅ test_protein_decoder_constant_dispersion passed!")

def test_protein_decoder_stability():
    torch.manual_seed(42);B,D,P=16,30,8
    decoder=ProteinDecoderZYAttention(n_latent=D,n_output_proteins=P,n_batch=2,dispersion="protein-cell")
    z=torch.randn(B,D);logbeta=torch.randn(B,P);batch_index=torch.randint(0,2,(B,))
    _,rate_fore,mix_logit,_,_=decoder(z,batch_index,logbeta)
    assert torch.all(torch.isfinite(rate_fore)) and torch.all(torch.isfinite(mix_logit))
    print("✅ test_protein_decoder_stability passed!")

def test_protein_decoder_s_dimension():
    S,B,D_latent,N_batch,P=3,10,32,3,5
    decoder=ProteinDecoderZYAttention(n_latent=D_latent,n_output_proteins=P,n_batch=N_batch,dispersion="protein-cell")
    z_sb=torch.randn(S,B,D_latent);logbeta_sb=torch.randn(S,B,P);batch_index_b=torch.randint(0,N_batch,(B,))
    rate_back,rate_fore,mix_logits,_,py_r=decoder(z_sb,batch_index_b,logbeta_sb)
    assert rate_back.shape==(S,B,P) and rate_fore.shape==(S,B,P) and mix_logits.shape==(S,B,P) and py_r.shape==(S,B,P)
    assert torch.all(rate_fore > rate_back)
    print("✅ ProteinDecoderZYAttention S-dimension test passed!")

def test_protein_decoder_batch_effect():
    B,D_latent,N_batch,P=10,32,3,5
    decoder=ProteinDecoderZYAttention(n_latent=D_latent,n_output_proteins=P,n_batch=N_batch,dispersion="protein-cell")
    z=torch.randn(B,D_latent);logbeta=torch.randn(B,P);batch_index=torch.randint(0,N_batch,(B,))
    _,rate_fore1,mix_logits1,_,_=decoder(z,batch_index,logbeta)
    _,rate_fore2,mix_logits2,_,_=decoder(z,(batch_index+1)%N_batch,logbeta)
    assert not torch.allclose(rate_fore1,rate_fore2) and not torch.allclose(mix_logits1,mix_logits2)
    print("✅ ProteinDecoderZYAttention batch effect test passed!")

def test_protein_decoder_zy_attention_dropout():
    torch.manual_seed(0)
    decoder=ProteinDecoderZYAttention(n_latent=32,n_output_proteins=5,n_batch=3,dispersion="protein-cell",dropout_rate_attn=0.5)
    z=torch.randn(10,32);logbeta=torch.randn(10,5);batch_index=torch.randint(0,3,(10,))
    decoder.eval();_,rate_fore_eval,_,_,_=decoder(z,batch_index,logbeta)
    decoder.train();_,rate_fore_train1,_,_,_=decoder(z,batch_index,logbeta);_,rate_fore_train2,_,_,_=decoder(z,batch_index,logbeta)
    assert not torch.allclose(rate_fore_eval,rate_fore_train1)
    assert not torch.allclose(rate_fore_train1,rate_fore_train2)
    has_dropout=any(isinstance(m,nn.Dropout) for m in decoder.mlp.modules())
    assert has_dropout
    print("✅ ProteinDecoderZYAttention dropout test passed.")