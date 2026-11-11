package tensor

import "fmt"

func Fused1619Log2MatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Log10()
	return r, nil
}
func Fused1619Log2MatmulLog10Name() string { return "log2_matmul_log10" }
