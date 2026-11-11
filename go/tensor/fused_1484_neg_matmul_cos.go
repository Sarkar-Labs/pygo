package tensor

import "fmt"

func Fused1484NegMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Cos()
	return r, nil
}
func Fused1484NegMatmulCosName() string { return "neg_matmul_cos" }
