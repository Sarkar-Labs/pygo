package tensor

import "fmt"

func Fused1524NegMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Log2()
	return r, nil
}
func Fused1524NegMatmulLog2Name() string { return "neg_matmul_log2" }
