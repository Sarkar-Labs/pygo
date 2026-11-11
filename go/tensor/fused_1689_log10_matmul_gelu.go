package tensor

import "fmt"

func Fused1689Log10MatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.GELU()
	return r, nil
}
func Fused1689Log10MatmulGeluName() string { return "log10_matmul_gelu" }
