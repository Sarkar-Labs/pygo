package tensor

import "fmt"

func Fused1659Log10MatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Sin()
	return r, nil
}
func Fused1659Log10MatmulSinName() string { return "log10_matmul_sin" }
