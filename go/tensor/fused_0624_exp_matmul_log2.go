package tensor

import "fmt"

func Fused0624ExpMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Log2()
	return r, nil
}
func Fused0624ExpMatmulLog2Name() string { return "exp_matmul_log2" }
