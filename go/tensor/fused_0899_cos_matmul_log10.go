package tensor

import "fmt"

func Fused0899CosMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Log10()
	return r, nil
}
func Fused0899CosMatmulLog10Name() string { return "cos_matmul_log10" }
