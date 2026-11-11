package tensor

import "fmt"

func Fused1664Log10MatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Cos()
	return r, nil
}
func Fused1664Log10MatmulCosName() string { return "log10_matmul_cos" }
