package tensor

import "fmt"

func Fused0801SinSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Log2()
	return r, nil
}
func Fused0801SinSubLog2Name() string { return "sin_sub_log2" }
