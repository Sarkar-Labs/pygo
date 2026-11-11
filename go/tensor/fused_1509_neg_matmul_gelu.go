package tensor

import "fmt"

func Fused1509NegMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.GELU()
	return r, nil
}
func Fused1509NegMatmulGeluName() string { return "neg_matmul_gelu" }
