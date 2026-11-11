package tensor

import "fmt"

func Fused0539SqrtMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Log10()
	return r, nil
}
func Fused0539SqrtMatmulLog10Name() string { return "sqrt_matmul_log10" }
