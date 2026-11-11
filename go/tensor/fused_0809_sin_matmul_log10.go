package tensor

import "fmt"

func Fused0809SinMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Log10()
	return r, nil
}
func Fused0809SinMatmulLog10Name() string { return "sin_matmul_log10" }
